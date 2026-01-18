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
def get_single_entry_point(self) -> Optional['EntryPoint']:
    """Returns the first entrypoint for the project, or None if no entry point was provided because a docker image was provided."""
    if not self._entry_point:
        if not self.docker_image:
            raise LaunchError('Project must have at least one entry point unless docker image is specified.')
        return None
    return self._entry_point