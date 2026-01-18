import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
def read_config_file(module):
    """Reads all UPPERCASE variables defined in the given module file."""
    settings = importlib.import_module(module)
    return dict([(k, v) for k, v in settings.__dict__.items() if k.upper() == k])