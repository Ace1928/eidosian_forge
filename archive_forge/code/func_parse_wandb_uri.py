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
def parse_wandb_uri(uri: str) -> Tuple[str, str, str]:
    """Parse wandb uri to retrieve entity, project and run name."""
    ref = WandbReference.parse(uri)
    if not ref or not ref.entity or (not ref.project) or (not ref.run_id):
        raise LaunchError(f'Trouble parsing wandb uri {uri}')
    return (ref.entity, ref.project, ref.run_id)