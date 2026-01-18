import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def construct_launch_spec(uri: Optional[str], job: Optional[str], api: Api, name: Optional[str], project: Optional[str], entity: Optional[str], docker_image: Optional[str], resource: Optional[str], entry_point: Optional[List[str]], version: Optional[str], resource_args: Optional[Dict[str, Any]], launch_config: Optional[Dict[str, Any]], run_id: Optional[str], repository: Optional[str], author: Optional[str], sweep_id: Optional[str]=None) -> Dict[str, Any]:
    """Construct the launch specification from CLI arguments."""
    launch_spec = launch_config if launch_config is not None else {}
    if uri is not None:
        launch_spec['uri'] = uri
    if job is not None:
        launch_spec['job'] = job
    project, entity = set_project_entity_defaults(uri, job, api, project, entity, launch_config)
    launch_spec['entity'] = entity
    if author:
        launch_spec['author'] = author
    launch_spec['project'] = project
    if name:
        launch_spec['name'] = name
    if 'docker' not in launch_spec:
        launch_spec['docker'] = {}
    if docker_image:
        launch_spec['docker']['docker_image'] = docker_image
    if sweep_id:
        launch_spec['sweep_id'] = sweep_id
    if 'resource' not in launch_spec:
        launch_spec['resource'] = resource if resource else None
    if 'git' not in launch_spec:
        launch_spec['git'] = {}
    if version:
        launch_spec['git']['version'] = version
    if 'overrides' not in launch_spec:
        launch_spec['overrides'] = {}
    if not isinstance(launch_spec['overrides'].get('args', []), list):
        raise LaunchError('override args must be a list of strings')
    if resource_args:
        launch_spec['resource_args'] = resource_args
    if entry_point:
        launch_spec['overrides']['entry_point'] = entry_point
    if run_id is not None:
        launch_spec['run_id'] = run_id
    if repository:
        launch_config = launch_config or {}
        if launch_config.get('registry'):
            launch_config['registry']['url'] = repository
        else:
            launch_config['registry'] = {'url': repository}
    strip_resource_args_and_template_vars(launch_spec)
    return launch_spec