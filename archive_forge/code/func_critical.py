import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
    self._log.append((logging.CRITICAL, msg, args, kwargs))