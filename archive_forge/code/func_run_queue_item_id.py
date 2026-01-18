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
@run_queue_item_id.setter
def run_queue_item_id(self, value: str) -> None:
    self._run_queue_item_id = value