import _thread as thread
import atexit
import functools
import glob
import json
import logging
import numbers
import os
import re
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from types import TracebackType
from typing import (
import requests
import wandb
import wandb.env
from wandb import errors, trigger
from wandb._globals import _datatypes_set_callback
from wandb.apis import internal, public
from wandb.apis.internal import Api
from wandb.apis.public import Api as PublicApi
from wandb.proto.wandb_internal_pb2 import (
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal import job_builder
from wandb.sdk.lib.import_hooks import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath
from wandb.util import (
from wandb.viz import CustomChart, Visualize, custom_chart
from . import wandb_config, wandb_metric, wandb_summary
from .data_types._dtypes import TypeRegistry
from .interface.interface import GlobStr, InterfaceBase
from .interface.summary_record import SummaryRecord
from .lib import (
from .lib.exit_hooks import ExitHooks
from .lib.gitlib import GitRepo
from .lib.mailbox import MailboxError, MailboxHandle, MailboxProbe, MailboxProgress
from .lib.printer import get_printer
from .lib.proto_util import message_to_dict
from .lib.reporting import Reporter
from .lib.wburls import wburls
from .wandb_settings import Settings
from .wandb_setup import _WandbSetup
@_run_decorator._noop_on_finish()
@_run_decorator._attach
def alert(self, title: str, text: str, level: Optional[Union[str, 'AlertLevel']]=None, wait_duration: Union[int, float, timedelta, None]=None) -> None:
    """Launch an alert with the given title and text.

        Arguments:
            title: (str) The title of the alert, must be less than 64 characters long.
            text: (str) The text body of the alert.
            level: (str or wandb.AlertLevel, optional) The alert level to use, either: `INFO`, `WARN`, or `ERROR`.
            wait_duration: (int, float, or timedelta, optional) The time to wait (in seconds) before sending another
                alert with this title.
        """
    level = level or wandb.AlertLevel.INFO
    level_str: str = level.value if isinstance(level, wandb.AlertLevel) else level
    if level_str not in {lev.value for lev in wandb.AlertLevel}:
        raise ValueError("level must be one of 'INFO', 'WARN', or 'ERROR'")
    wait_duration = wait_duration or timedelta(minutes=1)
    if isinstance(wait_duration, int) or isinstance(wait_duration, float):
        wait_duration = timedelta(seconds=wait_duration)
    elif not callable(getattr(wait_duration, 'total_seconds', None)):
        raise ValueError('wait_duration must be an int, float, or datetime.timedelta')
    wait_duration = int(wait_duration.total_seconds() * 1000)
    if self._backend and self._backend.interface:
        self._backend.interface.publish_alert(title, text, level_str, wait_duration)