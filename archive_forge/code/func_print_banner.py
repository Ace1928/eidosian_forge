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
def print_banner(app, ssl):
    if not options.unix_socket:
        if options.url_prefix:
            prefix_str = f'/{options.url_prefix}/'
        else:
            prefix_str = ''
        logger.info('Visit me at http%s://%s:%s%s', 's' if ssl else '', options.address or '0.0.0.0', options.port, prefix_str)
    else:
        logger.info('Visit me via unix socket file: %s', options.unix_socket)
    logger.info('Broker: %s', app.connection().as_uri())
    logger.info('Registered tasks: \n%s', pformat(sorted(app.tasks.keys())))
    logger.debug('Settings: %s', pformat(settings))