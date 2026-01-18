import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def load_launch_sweep_config(config: Optional[str]) -> Any:
    if not config:
        return {}
    parsed_config = util.load_json_yaml_dict(config)
    if parsed_config is None:
        raise LaunchError(f'Could not load config from {config}. Check formatting')
    return parsed_config