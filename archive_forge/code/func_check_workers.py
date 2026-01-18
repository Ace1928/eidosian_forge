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
def check_workers(self, respawn: bool=True) -> None:
    """
        Check whether workers are still alive
        """
    self.log.debug('Checking worker processes')
    self.reap_workers()
    if respawn and self.status != self.Status.STOPPED:
        delta = self.num_workers - len(self.worker_dict)
        if delta:
            for i in range(delta):
                self.start_worker(burst=self._burst, _sleep=self._sleep)