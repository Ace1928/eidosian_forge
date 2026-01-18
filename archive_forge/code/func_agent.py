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
@cli.command(context_settings=CONTEXT, help='Run the W&B agent')
@click.pass_context
@click.option('--project', '-p', default=None, help="The name of the project where W&B runs created from the sweep are sent to. If the project is not specified, the run is sent to a project labeled 'Uncategorized'.")
@click.option('--entity', '-e', default=None, help="The username or team name where you want to send W&B runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an entity, the run will be sent to your default entity, which is usually your username.")
@click.option('--count', default=None, type=int, help='The max number of runs for this agent.')
@click.argument('sweep_id')
@display_error
def agent(ctx, project, entity, count, sweep_id):
    api = _get_cling_api()
    if api.api_key is None:
        wandb.termlog('Login to W&B to use the sweep agent feature')
        ctx.invoke(login, no_offline=True)
        api = _get_cling_api(reset=True)
    wandb.termlog('Starting wandb agent 🕵️')
    wandb_agent.agent(sweep_id, entity=entity, project=project, count=count)