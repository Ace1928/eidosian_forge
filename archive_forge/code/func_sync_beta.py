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
@beta.command(name='sync', context_settings=CONTEXT, help='Upload a training run to W&B')
@click.pass_context
@click.argument('wandb_dir', nargs=1, type=click.Path(exists=True))
@click.option('--id', 'run_id', help='The run you want to upload to.')
@click.option('--project', '-p', help='The project you want to upload to.')
@click.option('--entity', '-e', help='The entity to scope to.')
@click.option('--skip-console', is_flag=True, default=False, help='Skip console logs')
@click.option('--append', is_flag=True, default=False, help='Append run')
@click.option('--include', '-i', help='Glob to include. Can be used multiple times.', multiple=True)
@click.option('--exclude', '-e', help='Glob to exclude. Can be used multiple times.', multiple=True)
@click.option('--mark-synced/--no-mark-synced', is_flag=True, default=True, help='Mark runs as synced')
@click.option('--skip-synced/--no-skip-synced', is_flag=True, default=True, help='Skip synced runs')
@click.option('--dry-run', is_flag=True, help='Perform a dry run without uploading anything.')
@display_error
def sync_beta(ctx, wandb_dir=None, run_id: Optional[str]=None, project: Optional[str]=None, entity: Optional[str]=None, skip_console: bool=False, append: bool=False, include: Optional[str]=None, exclude: Optional[str]=None, skip_synced: bool=True, mark_synced: bool=True, dry_run: bool=False):
    import concurrent.futures
    from multiprocessing import cpu_count
    paths = set()
    if include:
        for pattern in include:
            matching_dirs = list(pathlib.Path(wandb_dir).glob(pattern))
            for d in matching_dirs:
                if not d.is_dir():
                    continue
                wandb_files = [p for p in d.glob('*.wandb') if p.is_file()]
                if len(wandb_files) > 1:
                    print(f'Multiple wandb files found in directory {d}, skipping')
                elif len(wandb_files) == 1:
                    paths.add(d)
    else:
        paths.update({p.parent for p in pathlib.Path(wandb_dir).glob('**/*.wandb')})
    for pattern in exclude:
        matching_dirs = list(pathlib.Path(wandb_dir).glob(pattern))
        for d in matching_dirs:
            if not d.is_dir():
                continue
            if d in paths:
                paths.remove(d)
    if skip_synced:
        synced_paths = set()
        for path in paths:
            wandb_synced_files = [p for p in path.glob('*.wandb.synced') if p.is_file()]
            if len(wandb_synced_files) > 1:
                print(f'Multiple wandb.synced files found in directory {path}, skipping')
            elif len(wandb_synced_files) == 1:
                synced_paths.add(path)
        paths -= synced_paths
    if run_id and len(paths) > 1:
        click.echo('id can only be set for a single run.', err=True)
        sys.exit(1)
    if not paths:
        click.echo('No runs to sync.')
        return
    click.echo('Found runs:')
    for path in paths:
        click.echo(f'  {path}')
    if dry_run:
        return
    wandb.sdk.wandb_setup.setup()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(paths), cpu_count())) as executor:
        futures = []
        for path in paths:
            wandb_file = [p for p in path.glob('*.wandb') if p.is_file()][0]
            future = executor.submit(wandb._sync, wandb_file, run_id=run_id, project=project, entity=entity, skip_console=skip_console, append=append, mark_synced=mark_synced)
            futures.append(future)
        for _ in concurrent.futures.as_completed(futures):
            pass