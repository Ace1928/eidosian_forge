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
@cli.command(context_settings=CONTEXT, no_args_is_help=True, help='Run a W&B launch sweep (Experimental).')
@click.option('--queue', '-q', default=None, help='The name of a queue to push the sweep to')
@click.option('--project', '-p', default=None, help='Name of the project which the agent will watch. If passed in, will override the project value passed in using a config file')
@click.option('--entity', '-e', default=None, help='The entity to use. Defaults to current logged-in user')
@click.option('--resume_id', '-r', default=None, help='Resume a launch sweep by passing an 8-char sweep id. Queue required')
@click.argument('config', required=False, type=click.Path(exists=True))
@click.pass_context
@display_error
def launch_sweep(ctx, project, entity, queue, config, resume_id):
    api = _get_cling_api()
    env = os.environ
    if api.api_key is None:
        wandb.termlog('Login to W&B to use the sweep feature')
        ctx.invoke(login, no_offline=True)
        api = _get_cling_api(reset=True)
    entity = entity or env.get('WANDB_ENTITY') or api.settings('entity')
    if entity is None:
        wandb.termerror('Must specify entity when using launch')
        return
    project = project or env.get('WANDB_PROJECT') or api.settings('project')
    if project is None:
        wandb.termerror('A project must be configured when using launch')
        return
    author = api.viewer().get('username') or entity
    if not (config or resume_id):
        wandb.termerror("'config' and/or 'resume_id' required")
        return
    parsed_user_config = sweep_utils.load_launch_sweep_config(config)
    launch_args: Dict[str, Any] = parsed_user_config.pop('launch', {})
    scheduler_args: Dict[str, Any] = parsed_user_config.pop('scheduler', {})
    settings: Dict[str, Any] = scheduler_args.pop('settings', {})
    scheduler_job: Optional[str] = scheduler_args.get('job')
    if scheduler_job:
        wandb.termwarn('Using a scheduler job for launch sweeps is *experimental* and may change without warning')
    queue: Optional[str] = queue or launch_args.get('queue')
    sweep_config, sweep_obj_id = (None, None)
    if not resume_id:
        sweep_config = parsed_user_config
        method = sweep_config.get('method')
        if scheduler_job and (not method):
            sweep_config['method'] = 'custom'
        elif scheduler_job and method != 'custom':
            wandb.termwarn("Use 'method': 'custom' in the sweep config when using scheduler jobs, or omit it entirely. For jobs using the wandb optimization engine (WandbScheduler), set the method in the sweep config under scheduler.settings.method ")
            settings['method'] = method
        if settings.get('method'):
            sweep_config['method'] = settings['method']
    else:
        found = api.sweep(resume_id, '{}', entity=entity, project=project)
        if not found:
            wandb.termerror(f'Could not find sweep {entity}/{project}/{resume_id}')
            return
        if found.get('state') == 'RUNNING':
            wandb.termerror(f'Cannot resume sweep {entity}/{project}/{resume_id}, it is already running')
            return
        sweep_obj_id = found['id']
        sweep_config = yaml.safe_load(found['config'])
        wandb.termlog(f'Resuming from existing sweep {entity}/{project}/{resume_id}')
        if len(parsed_user_config.keys()) > 0:
            wandb.termwarn('Sweep parameters loaded from resumed sweep, ignoring provided config')
        prev_scheduler = json.loads(found.get('scheduler') or '{}')
        run_spec = json.loads(prev_scheduler.get('run_spec', '{}'))
        if scheduler_job and run_spec.get('job') and (run_spec.get('job') != scheduler_job):
            wandb.termerror(f'Resuming a launch sweep with a different scheduler job is not supported. Job loaded from sweep: {run_spec.get('job')}, job in config: {scheduler_job}')
            return
        prev_scheduler_args, prev_settings = sweep_utils.get_previous_args(run_spec)
        scheduler_args.update(prev_scheduler_args)
        settings.update(prev_settings)
    if not queue:
        wandb.termerror("Launch-sweeps require setting a 'queue', use --queue option or a 'queue' key in the 'launch' section in the config")
        return
    entrypoint = Scheduler.ENTRYPOINT if not scheduler_job else None
    args = sweep_utils.construct_scheduler_args(return_job=scheduler_job is not None, sweep_config=sweep_config, queue=queue, project=project, author=author)
    if not args:
        return
    if not sweep_utils.check_job_exists(PublicApi(), sweep_config.get('job')):
        return False
    if not sweep_utils.check_job_exists(PublicApi(), scheduler_job):
        return False
    overrides = {'run_config': {}}
    if launch_args:
        overrides['run_config']['launch'] = launch_args
    if scheduler_args:
        overrides['run_config']['scheduler'] = scheduler_args
    if settings:
        overrides['run_config']['settings'] = settings
    if scheduler_job:
        overrides['run_config']['sweep_args'] = args
    else:
        overrides['args'] = args
    resource = scheduler_args.get('resource')
    if resource:
        if resource == 'local-process' and scheduler_job:
            wandb.termerror("Scheduler jobs cannot be run with the 'local-process' resource")
            return
        if resource == 'local-process' and scheduler_args.get('docker_image'):
            wandb.termerror("Scheduler jobs cannot be run with the 'local-process' resource and a docker image")
            return
    else:
        resource = 'local-process' if not scheduler_job else 'local-container'
    launch_scheduler_spec = launch_utils.construct_launch_spec(uri=Scheduler.PLACEHOLDER_URI, api=api, name='Scheduler.WANDB_SWEEP_ID', project=project, entity=entity, docker_image=scheduler_args.get('docker_image'), resource=resource, entry_point=entrypoint, resource_args=scheduler_args.get('resource_args', {}), repository=launch_args.get('registry', {}).get('url', None), job=scheduler_job, version=None, launch_config={'overrides': overrides}, run_id='WANDB_SWEEP_ID', author=None)
    launch_scheduler_with_queue = json.dumps({'queue': queue, 'run_queue_project': launch_utils.LAUNCH_DEFAULT_PROJECT, 'run_spec': json.dumps(launch_scheduler_spec)})
    sweep_id, warnings = api.upsert_sweep(sweep_config, project=project, entity=entity, obj_id=sweep_obj_id, launch_scheduler=launch_scheduler_with_queue, state='PENDING')
    sweep_utils.handle_sweep_config_violations(warnings)
    styled_id = click.style(sweep_id, fg='yellow')
    wandb.termlog(f'{('Resumed' if resume_id else 'Created')} sweep with ID: {styled_id}')
    sweep_url = wandb_sdk.wandb_sweep._get_sweep_url(api, sweep_id)
    if sweep_url:
        styled_url = click.style(sweep_url, underline=True, fg='blue')
        wandb.termlog(f'View sweep at: {styled_url}')
    wandb.termlog(f'Scheduler added to launch queue ({queue})')