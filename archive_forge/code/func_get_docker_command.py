import asyncio
import logging
import os
import shlex
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from .._project_spec import LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
def get_docker_command(image: str, env_vars: Dict[str, str], entry_cmd: Optional[List[str]]=None, docker_args: Optional[Dict[str, Any]]=None, additional_args: Optional[List[str]]=None) -> List[str]:
    """Construct the docker command using the image and docker args.

    Arguments:
    image: a Docker image to be run
    env_vars: a dictionary of environment variables for the command
    entry_cmd: the entry point command to run
    docker_args: a dictionary of additional docker args for the command
    """
    docker_path = 'docker'
    cmd: List[Any] = [docker_path, 'run', '--rm']
    for env_key, env_value in env_vars.items():
        cmd += ['-e', f'{shlex.quote(env_key)}={shlex.quote(env_value)}']
    if docker_args:
        for name, value in docker_args.items():
            if len(name) == 1:
                prefix = '-' + shlex.quote(name)
            else:
                prefix = '--' + shlex.quote(name)
            if isinstance(value, list):
                for v in value:
                    cmd += [prefix, shlex.quote(str(v))]
            elif isinstance(value, bool) and value:
                cmd += [prefix]
            else:
                cmd += [prefix, shlex.quote(str(value))]
    if entry_cmd:
        cmd += ['--entrypoint', entry_cmd[0]]
    cmd += [shlex.quote(image)]
    if entry_cmd and len(entry_cmd) > 1:
        cmd += entry_cmd[1:]
    if additional_args:
        cmd += additional_args
    return cmd