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
def load_wandb_config() -> Config:
    """Load wandb config from WANDB_CONFIG environment variable(s).

    The WANDB_CONFIG environment variable is a json string that can contain
    multiple config keys. The WANDB_CONFIG_[0-9]+ environment variables are
    used for environments where there is a limit on the length of environment
    variables. In that case, we shard the contents of WANDB_CONFIG into
    multiple environment variables numbered from 0.

    Returns:
        A dictionary of wandb config values.
    """
    config_str = os.environ.get('WANDB_CONFIG')
    if config_str is None:
        config_str = ''
        idx = 0
        while True:
            chunk = os.environ.get(f'WANDB_CONFIG_{idx}')
            if chunk is None:
                break
            config_str += chunk
            idx += 1
        if idx < 1:
            raise LaunchError('No WANDB_CONFIG or WANDB_CONFIG_[0-9]+ environment variables found')
    wandb_config = Config()
    try:
        env_config = json.loads(config_str)
    except json.JSONDecodeError as e:
        raise LaunchError(f'Failed to parse WANDB_CONFIG: {e}') from e
    wandb_config.update(env_config)
    return wandb_config