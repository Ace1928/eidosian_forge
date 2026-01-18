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
def fetch_and_validate_template_variables(runqueue: Any, fields: dict) -> Dict[str, Any]:
    template_variables = {}
    variable_schemas = {}
    for tv in runqueue.template_variables:
        variable_schemas[tv['name']] = json.loads(tv['schema'])
    for field in fields:
        field_parts = field.split('=')
        if len(field_parts) != 2:
            raise LaunchError(f'--set-var value must be in the format "--set-var key1=value1", instead got: {field}')
        key, val = field_parts
        if key not in variable_schemas:
            raise LaunchError(f'Queue {runqueue.name} does not support overriding {key}.')
        schema = variable_schemas.get(key, {})
        field_type = schema.get('type')
        try:
            if field_type == 'integer':
                val = int(val)
            elif field_type == 'number':
                val = float(val)
        except ValueError:
            raise LaunchError(f'Value for {key} must be of type {field_type}.')
        template_variables[key] = val
    return template_variables