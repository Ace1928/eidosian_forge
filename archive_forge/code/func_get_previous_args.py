import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def get_previous_args(run_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse through previous scheduler run_spec.

    returns scheduler_args and settings.
    """
    scheduler_args = run_spec.get('overrides', {}).get('run_config', {}).get('scheduler', {})
    if run_spec.get('resource'):
        scheduler_args['resource'] = run_spec['resource']
    if run_spec.get('resource_args'):
        scheduler_args['resource_args'] = run_spec['resource_args']
    settings = run_spec.get('overrides', {}).get('run_config', {}).get('settings', {})
    return (scheduler_args, settings)