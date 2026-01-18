import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def compute_command(self, user_parameters: Optional[List[str]]) -> List[str]:
    """Converts user parameter dictionary to a string."""
    ret = self.command
    if user_parameters:
        return ret + user_parameters
    return ret