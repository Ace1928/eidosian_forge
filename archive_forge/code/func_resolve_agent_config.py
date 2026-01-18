import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml
import wandb
from wandb.apis.internal import Api
from . import loader
from ._project_spec import LaunchProject
from .agent import LaunchAgent
from .builder.build import construct_agent_configs
from .environment.local_environment import LocalEnvironment
from .errors import ExecutionError, LaunchError
from .runner.abstract import AbstractRun
from .utils import (
def resolve_agent_config(entity: Optional[str], project: Optional[str], max_jobs: Optional[int], queues: Optional[Tuple[str]], config: Optional[str]) -> Tuple[Dict[str, Any], Api]:
    """Resolve the agent config.

    Arguments:
        api (Api): The api.
        entity (str): The entity.
        project (str): The project.
        max_jobs (int): The max number of jobs.
        queues (Tuple[str]): The queues.
        config (str): The config.

    Returns:
        Tuple[Dict[str, Any], Api]: The resolved config and api.
    """
    defaults = {'project': LAUNCH_DEFAULT_PROJECT, 'max_jobs': 1, 'max_schedulers': 1, 'queues': [], 'registry': {}, 'builder': {}}
    user_set_project = False
    resolved_config: Dict[str, Any] = defaults
    config_path = config or os.path.expanduser(LAUNCH_CONFIG_FILE)
    if os.path.isfile(config_path):
        launch_config = {}
        with open(config_path) as f:
            try:
                launch_config = yaml.safe_load(f)
                if launch_config is None:
                    launch_config = {}
            except yaml.YAMLError as e:
                raise LaunchError(f'Invalid launch agent config: {e}')
        if launch_config.get('project') is not None:
            user_set_project = True
        resolved_config.update(launch_config.items())
    elif config is not None:
        raise LaunchError(f'Could not find use specified launch config file: {config_path}')
    if os.environ.get('WANDB_PROJECT') is not None:
        resolved_config.update({'project': os.environ.get('WANDB_PROJECT')})
        user_set_project = True
    if os.environ.get('WANDB_ENTITY') is not None:
        resolved_config.update({'entity': os.environ.get('WANDB_ENTITY')})
    if os.environ.get('WANDB_LAUNCH_MAX_JOBS') is not None:
        resolved_config.update({'max_jobs': int(os.environ.get('WANDB_LAUNCH_MAX_JOBS', 1))})
    if project is not None:
        resolved_config.update({'project': project})
        user_set_project = True
    if entity is not None:
        resolved_config.update({'entity': entity})
    if max_jobs is not None:
        resolved_config.update({'max_jobs': int(max_jobs)})
    if queues:
        resolved_config.update({'queues': list(queues)})
    if resolved_config.get('queue'):
        if isinstance(resolved_config.get('queue'), str):
            resolved_config['queues'].append(resolved_config['queue'])
        else:
            raise LaunchError(f"Invalid launch agent config for key 'queue' with type: {type(resolved_config.get('queue'))}" + " (expected str). Specify multiple queues with the 'queues' key")
    keys = ['project', 'entity']
    settings = {k: resolved_config.get(k) for k in keys if resolved_config.get(k) is not None}
    api = Api(default_settings=settings)
    if resolved_config.get('entity') is None:
        resolved_config.update({'entity': api.default_entity})
    if user_set_project:
        wandb.termwarn('Specifying a project for the launch agent is deprecated. Please use queues found in the Launch application at https://wandb.ai/launch.')
    return (resolved_config, api)