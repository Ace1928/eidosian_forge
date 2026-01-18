import asyncio
import configparser
import datetime
import getpass
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from functools import wraps
from typing import Any, Dict, Optional
import click
import yaml
from click.exceptions import ClickException
from dockerpycreds.utils import find_executable
import wandb
import wandb.env
import wandb.sdk.verify.verify as wandb_verify
from wandb import Config, Error, env, util, wandb_agent, wandb_sdk
from wandb.apis import InternalApi, PublicApi
from wandb.apis.public import RunQueue
from wandb.integration.magic import magic_install
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.launch import utils as launch_utils
from wandb.sdk.launch._launch_add import _launch_add
from wandb.sdk.launch.errors import ExecutionError, LaunchError
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.wburls import wburls
from wandb.sync import SyncManager, get_run_from_path, get_runs
import __main__
@cli.command(context_settings=RUN_CONTEXT)
@click.pass_context
@click.argument('docker_run_args', nargs=-1)
@click.argument('docker_image', required=False)
@click.option('--nvidia/--no-nvidia', default=find_executable('nvidia-docker') is not None, help='Use the nvidia runtime, defaults to nvidia if nvidia-docker is present')
@click.option('--digest', is_flag=True, default=False, help='Output the image digest and exit')
@click.option('--jupyter/--no-jupyter', default=False, help='Run jupyter lab in the container')
@click.option('--dir', default='/app', help='Which directory to mount the code in the container')
@click.option('--no-dir', is_flag=True, help="Don't mount the current directory")
@click.option('--shell', default='/bin/bash', help='The shell to start the container with')
@click.option('--port', default='8888', help='The host port to bind jupyter on')
@click.option('--cmd', help='The command to run in the container')
@click.option('--no-tty', is_flag=True, default=False, help='Run the command without a tty')
@display_error
def docker(ctx, docker_run_args, docker_image, nvidia, digest, jupyter, dir, no_dir, shell, port, cmd, no_tty):
    """Run your code in a docker container.

    W&B docker lets you run your code in a docker image ensuring wandb is configured. It
    adds the WANDB_DOCKER and WANDB_API_KEY environment variables to your container and
    mounts the current directory in /app by default.  You can pass additional args which
    will be added to `docker run` before the image name is declared, we'll choose a
    default image for you if one isn't passed:

    ```sh
    wandb docker -v /mnt/dataset:/app/data
    wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter
    wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
    ```

    By default, we override the entrypoint to check for the existence of wandb and
    install it if not present.  If you pass the --jupyter flag we will ensure jupyter is
    installed and start jupyter lab on port 8888.  If we detect nvidia-docker on your
    system we will use the nvidia runtime.  If you just want wandb to set environment
    variable to an existing docker run command, see the wandb docker-run command.
    """
    api = InternalApi()
    if not find_executable('docker'):
        raise ClickException('Docker not installed, install it from https://docker.com')
    args = list(docker_run_args)
    image = docker_image or ''
    if len(args) > 0 and args[0] == 'run':
        args.pop(0)
    if image == '' and len(args) > 0:
        image = args.pop(0)
    if not util.docker_image_regex(image.split('@')[0]):
        if image:
            args = args + [image]
        image = wandb.docker.default_image(gpu=nvidia)
        subprocess.call(['docker', 'pull', image])
    _, repo_name, tag = wandb.docker.parse(image)
    resolved_image = wandb.docker.image_id(image)
    if resolved_image is None:
        raise ClickException("Couldn't find image locally or in a registry, try running `docker pull %s`" % image)
    if digest:
        sys.stdout.write(resolved_image)
        exit(0)
    existing = wandb.docker.shell(['ps', '-f', 'ancestor=%s' % resolved_image, '-q'])
    if existing:
        if click.confirm('Found running container with the same image, do you want to attach?'):
            subprocess.call(['docker', 'attach', existing.split('\n')[0]])
            exit(0)
    cwd = os.getcwd()
    command = ['docker', 'run', '-e', 'LANG=C.UTF-8', '-e', 'WANDB_DOCKER=%s' % resolved_image, '--ipc=host', '-v', wandb.docker.entrypoint + ':/wandb-entrypoint.sh', '--entrypoint', '/wandb-entrypoint.sh']
    if nvidia:
        command.extend(['--runtime', 'nvidia'])
    if not no_dir:
        command.extend(['-v', cwd + ':' + dir, '-w', dir])
    if api.api_key:
        command.extend(['-e', 'WANDB_API_KEY=%s' % api.api_key])
    else:
        wandb.termlog("Couldn't find WANDB_API_KEY, run `wandb login` to enable streaming metrics")
    if jupyter:
        command.extend(['-e', 'WANDB_ENSURE_JUPYTER=1', '-p', port + ':8888'])
        no_tty = True
        cmd = 'jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir %s' % dir
    command.extend(args)
    if no_tty:
        command.extend([image, shell, '-c', cmd])
    else:
        if cmd:
            command.extend(['-e', 'WANDB_COMMAND=%s' % cmd])
        command.extend(['-it', image, shell])
        wandb.termlog('Launching docker container 🚢')
    subprocess.call(command)