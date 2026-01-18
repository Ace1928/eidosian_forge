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
def get_redis_from_config(settings, connection_class=Redis):
    """Returns a StrictRedis instance from a dictionary of settings.
    To use redis sentinel, you must specify a dictionary in the configuration file.
    Example of a dictionary with keys without values:
    SENTINEL = {'INSTANCES':, 'SOCKET_TIMEOUT':, 'USERNAME':, 'PASSWORD':, 'DB':, 'MASTER_NAME':, 'SENTINEL_KWARGS':}
    """
    if settings.get('REDIS_URL') is not None:
        return connection_class.from_url(settings['REDIS_URL'])
    elif settings.get('SENTINEL') is not None:
        instances = settings['SENTINEL'].get('INSTANCES', [('localhost', 26379)])
        master_name = settings['SENTINEL'].get('MASTER_NAME', 'mymaster')
        connection_kwargs = {'db': settings['SENTINEL'].get('DB', 0), 'username': settings['SENTINEL'].get('USERNAME', None), 'password': settings['SENTINEL'].get('PASSWORD', None), 'socket_timeout': settings['SENTINEL'].get('SOCKET_TIMEOUT', None), 'ssl': settings['SENTINEL'].get('SSL', False)}
        connection_kwargs.update(settings['SENTINEL'].get('CONNECTION_KWARGS', {}))
        sentinel_kwargs = settings['SENTINEL'].get('SENTINEL_KWARGS', {})
        sn = Sentinel(instances, sentinel_kwargs=sentinel_kwargs, **connection_kwargs)
        return sn.master_for(master_name)
    ssl = settings.get('REDIS_SSL', False)
    if isinstance(ssl, str):
        if ssl.lower() in ['y', 'yes', 't', 'true']:
            ssl = True
        elif ssl.lower() in ['n', 'no', 'f', 'false', '']:
            ssl = False
        else:
            raise ValueError('REDIS_SSL is a boolean and must be "True" or "False".')
    kwargs = {'host': settings.get('REDIS_HOST', 'localhost'), 'port': settings.get('REDIS_PORT', 6379), 'db': settings.get('REDIS_DB', 0), 'password': settings.get('REDIS_PASSWORD', None), 'ssl': ssl, 'ssl_ca_certs': settings.get('REDIS_SSL_CA_CERTS', None), 'ssl_cert_reqs': settings.get('REDIS_SSL_CERT_REQS', 'required')}
    return connection_class(**kwargs)