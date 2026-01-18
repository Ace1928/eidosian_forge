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
def apply_patch(patch_string: str, dst_dir: str) -> None:
    """Applies a patch file to a directory."""
    _logger.info('Applying diff.patch')
    with open(os.path.join(dst_dir, 'diff.patch'), 'w') as fp:
        fp.write(patch_string)
    try:
        subprocess.check_call(['patch', '-s', f'--directory={dst_dir}', '-p1', '-i', 'diff.patch'])
    except subprocess.CalledProcessError:
        raise wandb.Error('Failed to apply diff.patch associated with run.')