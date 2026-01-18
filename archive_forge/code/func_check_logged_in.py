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
def check_logged_in(api: Api) -> bool:
    """Check if a user is logged in.

    Raises an error if the viewer doesn't load (likely a broken API key). Expected time
    cost is 0.1-0.2 seconds.
    """
    res = api.api.viewer()
    if not res:
        raise LaunchError('Could not connect with current API-key. Please relogin using `wandb login --relogin` and try again (see `wandb login --help` for more options)')
    return True