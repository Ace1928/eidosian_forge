import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def construct_agent_configs(launch_config: Optional[Dict]=None, build_config: Optional[Dict]=None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    registry_config = None
    environment_config = None
    if launch_config is not None:
        build_config = launch_config.get('builder')
        registry_config = launch_config.get('registry')
    default_launch_config = None
    if os.path.exists(os.path.expanduser(LAUNCH_CONFIG_FILE)):
        with open(os.path.expanduser(LAUNCH_CONFIG_FILE)) as f:
            default_launch_config = yaml.safe_load(f) or {}
        environment_config = default_launch_config.get('environment')
    build_config, registry_config = resolve_build_and_registry_config(default_launch_config, build_config, registry_config)
    return (environment_config, build_config, registry_config)