import contextlib
import errno
import logging
import math
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from enum import Enum
from random import shuffle
from types import FrameType
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from contextlib import suppress
import redis.exceptions
from . import worker_registration
from .command import PUBSUB_CHANNEL_TEMPLATE, handle_command, parse_payload
from .connections import get_current_connection, pop_connection, push_connection
from .defaults import (
from .exceptions import DequeueTimeout, DeserializationError, ShutDownImminentException
from .job import Job, JobStatus
from .logutils import blue, green, setup_loghandlers, yellow
from .maintenance import clean_intermediate_queue
from .queue import Queue
from .registry import StartedJobRegistry, clean_registries
from .scheduler import RQScheduler
from .serializers import resolve_serializer
from .suspension import is_suspended
from .timeouts import HorseMonitorTimeoutException, JobTimeoutException, UnixSignalDeathPenalty
from .utils import as_text, backend_class, compact, ensure_list, get_version, utcformat, utcnow, utcparse
from .version import VERSION
def kill_horse(self, sig: signal.Signals=SIGKILL):
    """Kill the horse but catch "No such process" error has the horse could already be dead.

        Args:
            sig (signal.Signals, optional): _description_. Defaults to SIGKILL.
        """
    try:
        os.killpg(os.getpgid(self.horse_pid), sig)
        self.log.info('Killed horse pid %s', self.horse_pid)
    except OSError as e:
        if e.errno == errno.ESRCH:
            self.log.debug('Horse already dead')
        else:
            raise