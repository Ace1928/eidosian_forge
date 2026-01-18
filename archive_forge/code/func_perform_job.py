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
def perform_job(self, job: 'Job', queue: 'Queue') -> bool:
    """Performs the actual work of a job.  Will/should only be called
        inside the work horse's process.

        Args:
            job (Job): The Job
            queue (Queue): The Queue

        Returns:
            bool: True after finished.
        """
    push_connection(self.connection)
    started_job_registry = queue.started_job_registry
    self.log.debug('Started Job Registry set.')
    try:
        remove_from_intermediate_queue = len(self.queues) == 1
        self.prepare_job_execution(job, remove_from_intermediate_queue)
        job.started_at = utcnow()
        timeout = job.timeout or self.queue_class.DEFAULT_TIMEOUT
        with self.death_penalty_class(timeout, JobTimeoutException, job_id=job.id):
            self.log.debug('Performing Job...')
            rv = job.perform()
            self.log.debug('Finished performing Job ID %s', job.id)
        job.ended_at = utcnow()
        job._result = rv
        job.heartbeat(utcnow(), job.success_callback_timeout)
        job.execute_success_callback(self.death_penalty_class, rv)
        self.handle_job_success(job=job, queue=queue, started_job_registry=started_job_registry)
    except:
        self.log.debug('Job %s raised an exception.', job.id)
        job.ended_at = utcnow()
        exc_info = sys.exc_info()
        exc_string = ''.join(traceback.format_exception(*exc_info))
        try:
            job.heartbeat(utcnow(), job.failure_callback_timeout)
            job.execute_failure_callback(self.death_penalty_class, *exc_info)
        except:
            exc_info = sys.exc_info()
            exc_string = ''.join(traceback.format_exception(*exc_info))
        self.handle_job_failure(job=job, exc_string=exc_string, queue=queue, started_job_registry=started_job_registry)
        self.handle_exception(job, *exc_info)
        return False
    finally:
        pop_connection()
    self.log.info('%s: %s (%s)', green(job.origin), blue('Job OK'), job.id)
    if rv is not None:
        self.log.debug('Result: %r', yellow(as_text(str(rv))))
    if self.log_result_lifespan:
        result_ttl = job.get_result_ttl(self.default_result_ttl)
        if result_ttl == 0:
            self.log.info('Result discarded immediately')
        elif result_ttl > 0:
            self.log.info('Result is kept for %s seconds', result_ttl)
        else:
            self.log.info('Result will never expire, clean up result key manually')
    return True