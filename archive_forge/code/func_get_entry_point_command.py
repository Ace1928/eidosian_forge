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
def get_entry_point_command(entry_point: Optional['EntryPoint'], parameters: List[str]) -> List[str]:
    """Returns the shell command to execute in order to run the specified entry point.

    Arguments:
    entry_point: Entry point to run
    parameters: Parameters (dictionary) for the entry point command

    Returns:
        List of strings representing the shell command to be executed
    """
    if entry_point is None:
        return []
    return entry_point.compute_command(parameters)