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
def handle_job_success(self, job: 'Job', queue: 'Queue', started_job_registry: StartedJobRegistry):
    """Handles the successful execution of certain job.
        It will remove the job from the `StartedJobRegistry`, adding it to the `SuccessfulJobRegistry`,
        and run a few maintenance tasks including:
            - Resting the current job ID
            - Enqueue dependents
            - Incrementing the job count and working time
            - Handling of the job successful execution

        Runs within a loop with the `watch` method so that protects interactions
        with dependents keys.

        Args:
            job (Job): The job that was successful.
            queue (Queue): The queue
            started_job_registry (StartedJobRegistry): The started registry
        """
    self.log.debug('Handling successful execution of job %s', job.id)
    with self.connection.pipeline() as pipeline:
        while True:
            try:
                pipeline.watch(job.dependents_key)
                queue.enqueue_dependents(job, pipeline=pipeline)
                if not pipeline.explicit_transaction:
                    pipeline.multi()
                self.set_current_job_id(None, pipeline=pipeline)
                self.increment_successful_job_count(pipeline=pipeline)
                self.increment_total_working_time(job.ended_at - job.started_at, pipeline)
                result_ttl = job.get_result_ttl(self.default_result_ttl)
                if result_ttl != 0:
                    self.log.debug("Saving job %s's successful execution result", job.id)
                    job._handle_success(result_ttl, pipeline=pipeline)
                job.cleanup(result_ttl, pipeline=pipeline, remove_from_queue=False)
                self.log.debug('Removing job %s from StartedJobRegistry', job.id)
                started_job_registry.remove(job, pipeline=pipeline)
                pipeline.execute()
                self.log.debug('Finished handling successful execution of job %s', job.id)
                break
            except redis.exceptions.WatchError:
                continue