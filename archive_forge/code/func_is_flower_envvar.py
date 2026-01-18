import os
import sys
import atexit
import signal
import logging
from pprint import pformat
from logging import NullHandler
import click
from tornado.options import options
from tornado.options import parse_command_line, parse_config_file
from tornado.log import enable_pretty_logging
from celery.bin.base import CeleryCommand
from .app import Flower
from .urls import settings
from .utils import abs_path, prepend_url, strtobool
from .options import DEFAULT_CONFIG_FILE, default_options
from .views.auth import validate_auth_option
def is_flower_envvar(name):
    return name.startswith(ENV_VAR_PREFIX) and name[len(ENV_VAR_PREFIX):].lower() in default_options