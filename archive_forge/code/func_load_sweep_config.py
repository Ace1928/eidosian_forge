import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def load_sweep_config(sweep_config_path: str) -> Optional[Dict[str, Any]]:
    """Load a sweep yaml from path."""
    try:
        yaml_file = open(sweep_config_path)
    except OSError:
        wandb.termerror(f"Couldn't open sweep file: {sweep_config_path}")
        return None
    try:
        config: Optional[Dict[str, Any]] = yaml.safe_load(yaml_file)
    except yaml.YAMLError as err:
        wandb.termerror(f'Error in configuration file: {err}')
        return None
    if not config:
        wandb.termerror('Configuration file is empty')
        return None
    return config