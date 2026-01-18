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
def dequeue_job_and_maintain_ttl(self, timeout: Optional[int], max_idle_time: Optional[int]=None) -> Tuple['Job', 'Queue']:
    """Dequeues a job while maintaining the TTL.

        Returns:
            result (Tuple[Job, Queue]): A tuple with the job and the queue.
        """
    result = None
    qnames = ','.join(self.queue_names())
    self.set_state(WorkerStatus.IDLE)
    self.procline('Listening on ' + qnames)
    self.log.debug('*** Listening on %s...', green(qnames))
    connection_wait_time = 1.0
    idle_since = utcnow()
    idle_time_left = max_idle_time
    while True:
        try:
            self.heartbeat()
            if self.should_run_maintenance_tasks:
                self.run_maintenance_tasks()
            if timeout is not None and idle_time_left is not None:
                timeout = min(timeout, idle_time_left)
            self.log.debug('Dequeueing jobs on queues %s and timeout %s', green(qnames), timeout)
            result = self.queue_class.dequeue_any(self._ordered_queues, timeout, connection=self.connection, job_class=self.job_class, serializer=self.serializer, death_penalty_class=self.death_penalty_class)
            if result is not None:
                job, queue = result
                self.reorder_queues(reference_queue=queue)
                self.log.debug('Dequeued job %s from %s', blue(job.id), green(queue.name))
                job.redis_server_version = self.get_redis_server_version()
                if self.log_job_description:
                    self.log.info('%s: %s (%s)', green(queue.name), blue(job.description), job.id)
                else:
                    self.log.info('%s: %s', green(queue.name), job.id)
            break
        except DequeueTimeout:
            if max_idle_time is not None:
                idle_for = (utcnow() - idle_since).total_seconds()
                idle_time_left = math.ceil(max_idle_time - idle_for)
                if idle_time_left <= 0:
                    break
        except redis.exceptions.ConnectionError as conn_err:
            self.log.error('Could not connect to Redis instance: %s Retrying in %d seconds...', conn_err, connection_wait_time)
            time.sleep(connection_wait_time)
            connection_wait_time *= self.exponential_backoff_factor
            connection_wait_time = min(connection_wait_time, self.max_connection_wait_time)
        else:
            connection_wait_time = 1.0
    self.heartbeat()
    return result