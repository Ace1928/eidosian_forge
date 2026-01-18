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
def apply_env_options():
    """apply options passed through environment variables"""
    env_options = filter(is_flower_envvar, os.environ)
    for env_var_name in env_options:
        name = env_var_name.replace(ENV_VAR_PREFIX, '', 1).lower()
        value = os.environ[env_var_name]
        try:
            option = options._options[name]
        except KeyError:
            option = options._options[name.replace('_', '-')]
        if option.multiple:
            value = [option.type(i) for i in value.split(',')]
        elif option.type is bool:
            value = bool(strtobool(value))
        else:
            value = option.type(value)
        setattr(options, name, value)