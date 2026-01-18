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
def prepare_job_execution(self, job: 'Job', remove_from_intermediate_queue: bool=False):
    """Performs misc bookkeeping like updating states prior to
        job execution.
        """
    self.log.debug('Preparing for execution of Job ID %s', job.id)
    with self.connection.pipeline() as pipeline:
        self.set_current_job_id(job.id, pipeline=pipeline)
        self.set_current_job_working_time(0, pipeline=pipeline)
        heartbeat_ttl = self.get_heartbeat_ttl(job)
        self.heartbeat(heartbeat_ttl, pipeline=pipeline)
        job.heartbeat(utcnow(), heartbeat_ttl, pipeline=pipeline)
        job.prepare_for_execution(self.name, pipeline=pipeline)
        if remove_from_intermediate_queue:
            from .queue import Queue
            queue = Queue(job.origin, connection=self.connection)
            pipeline.lrem(queue.intermediate_queue_key, 1, job.id)
        pipeline.execute()
        self.log.debug('Job preparation finished.')
    msg = 'Processing {0} from {1} since {2}'
    self.procline(msg.format(job.func_name, job.origin, time.time()))