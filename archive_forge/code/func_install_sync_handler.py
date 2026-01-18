from __future__ import annotations
import numbers
import socket
import sys
from datetime import datetime
from signal import Signals
from types import FrameType
from typing import Any
from celery import VERSION_BANNER, Celery, beat, platforms
from celery.utils.imports import qualname
from celery.utils.log import LOG_LEVELS, get_logger
from celery.utils.time import humanize_seconds
def install_sync_handler(self, service: beat.Service) -> None:
    """Install a `SIGTERM` + `SIGINT` handler saving the schedule."""

    def _sync(signum: Signals, frame: FrameType) -> None:
        service.sync()
        raise SystemExit()
    platforms.signals.update(SIGTERM=_sync, SIGINT=_sync)