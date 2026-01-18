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
def fetch_project_diff(entity: str, project: str, run_name: str, api: Api) -> Optional[str]:
    """Fetches project diff from wandb servers."""
    _logger.info('Searching for diff.patch')
    patch = None
    try:
        _, _, patch, _ = api.run_config(project, run_name, entity)
    except CommError:
        pass
    return patch