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
def extract_settings():
    settings['debug'] = options.debug
    if options.cookie_secret:
        settings['cookie_secret'] = options.cookie_secret
    if options.url_prefix:
        for name in ['login_url', 'static_url_prefix']:
            settings[name] = prepend_url(settings[name], options.url_prefix)
    if options.auth:
        settings['oauth'] = {'key': options.oauth2_key or os.environ.get('FLOWER_OAUTH2_KEY'), 'secret': options.oauth2_secret or os.environ.get('FLOWER_OAUTH2_SECRET'), 'redirect_uri': options.oauth2_redirect_uri or os.environ.get('FLOWER_OAUTH2_REDIRECT_URI')}
    if options.certfile and options.keyfile:
        settings['ssl_options'] = dict(certfile=abs_path(options.certfile), keyfile=abs_path(options.keyfile))
        if options.ca_certs:
            settings['ssl_options']['ca_certs'] = abs_path(options.ca_certs)
    if options.auth and (not validate_auth_option(options.auth)):
        logger.error("Invalid '--auth' option: %s", options.auth)
        sys.exit(1)