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
def parse_function_args(arguments):
    args = []
    kwargs = {}
    for argument in arguments:
        keyword, value = parse_function_arg(argument, len(args) + 1)
        if keyword is not None:
            if keyword in kwargs:
                raise click.BadParameter("You can't specify multiple values for the same keyword.")
            kwargs[keyword] = value
        else:
            args.append(value)
    return (args, kwargs)