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
def resolve_build_and_registry_config(default_launch_config: Optional[Dict[str, Any]], build_config: Optional[Dict[str, Any]], registry_config: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved_build_config: Dict[str, Any] = {}
    if build_config is None and default_launch_config is not None:
        resolved_build_config = default_launch_config.get('builder', {})
    elif build_config is not None:
        resolved_build_config = build_config
    resolved_registry_config: Dict[str, Any] = {}
    if registry_config is None and default_launch_config is not None:
        resolved_registry_config = default_launch_config.get('registry', {})
    elif registry_config is not None:
        resolved_registry_config = registry_config
    validate_build_and_registry_configs(resolved_build_config, resolved_registry_config)
    return (resolved_build_config, resolved_registry_config)