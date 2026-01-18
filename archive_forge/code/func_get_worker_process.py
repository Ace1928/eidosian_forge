import contextlib
import errno
import logging
import os
import signal
import time
from enum import Enum
from multiprocessing import Process
from typing import Dict, List, NamedTuple, Optional, Type, Union
from uuid import uuid4
from redis import ConnectionPool, Redis
from rq.serializers import DefaultSerializer
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .utils import parse_names
from .worker import BaseWorker, Worker
def get_worker_process(self, name: str, burst: bool, _sleep: float=0, logging_level: str='INFO') -> Process:
    """Returns the worker process"""
    return Process(target=run_worker, args=(name, self._queue_names, self._connection_class, self._pool_class, self._pool_kwargs), kwargs={'_sleep': _sleep, 'burst': burst, 'logging_level': logging_level, 'worker_class': self.worker_class, 'job_class': self.job_class, 'serializer': self.serializer}, name=f'Worker {name} (WorkerPool {self.name})')