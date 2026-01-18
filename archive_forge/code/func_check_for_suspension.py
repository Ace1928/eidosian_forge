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
def check_for_suspension(self, burst: bool):
    """Check to see if workers have been suspended by `rq suspend`"""
    before_state = None
    notified = False
    while not self._stop_requested and is_suspended(self.connection, self):
        if burst:
            self.log.info('Suspended in burst mode, exiting')
            self.log.info('Note: There could still be unfinished jobs on the queue')
            raise StopRequested
        if not notified:
            self.log.info('Worker suspended, run `rq resume` to resume')
            before_state = self.get_state()
            self.set_state(WorkerStatus.SUSPENDED)
            notified = True
        time.sleep(1)
    if before_state:
        self.set_state(before_state)