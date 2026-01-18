import logging
import os
import signal
import time
import traceback
from datetime import datetime
from enum import Enum
from multiprocessing import Process
from typing import List, Set
from redis import ConnectionPool, Redis
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, DEFAULT_SCHEDULER_FALLBACK_PERIOD
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .registry import ScheduledJobRegistry
from .serializers import resolve_serializer
from .utils import current_timestamp, parse_names
def enqueue_scheduled_jobs(self):
    """Enqueue jobs whose timestamp is in the past"""
    self._status = self.Status.WORKING
    if not self._scheduled_job_registries and self._acquired_locks:
        self.prepare_registries()
    for registry in self._scheduled_job_registries:
        timestamp = current_timestamp()
        job_ids = registry.get_jobs_to_schedule(timestamp)
        if not job_ids:
            continue
        queue = Queue(registry.name, connection=self.connection, serializer=self.serializer)
        with self.connection.pipeline() as pipeline:
            jobs = Job.fetch_many(job_ids, connection=self.connection, serializer=self.serializer)
            for job in jobs:
                if job is not None:
                    queue._enqueue_job(job, pipeline=pipeline, at_front=bool(job.enqueue_at_front))
                    registry.remove(job, pipeline=pipeline)
            pipeline.execute()
    self._status = self.Status.STARTED