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
@classmethod
def get_locking_key(cls, name: str):
    """Returns scheduler key for a given queue name"""
    return SCHEDULER_LOCKING_KEY_TEMPLATE % name