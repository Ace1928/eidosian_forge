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
def state_symbol(state):
    symbols = {WorkerStatus.BUSY: red('busy'), WorkerStatus.IDLE: green('idle'), WorkerStatus.SUSPENDED: yellow('suspended')}
    try:
        return symbols[state]
    except KeyError:
        return state