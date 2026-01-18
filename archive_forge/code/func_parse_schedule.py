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
def parse_schedule(schedule_in, schedule_at):
    if schedule_in is not None:
        if schedule_at is not None:
            raise click.BadArgumentUsage("You can't specify both --schedule-in and --schedule-at")
        return datetime.now(timezone.utc) + timedelta(seconds=parse_timeout(schedule_in))
    elif schedule_at is not None:
        return datetime.strptime(schedule_at, '%Y-%m-%dT%H:%M:%S')