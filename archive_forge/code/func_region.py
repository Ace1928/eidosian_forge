import logging
import os
from typing import Dict, Optional
from wandb.sdk.launch.errors import LaunchError
from wandb.util import get_module
from ..utils import S3_URI_RE, event_loop_thread_exec
from .abstract import AbstractEnvironment
@region.setter
def region(self, region: str) -> None:
    self._region = region