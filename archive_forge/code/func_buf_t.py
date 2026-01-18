import threading
from celery import states
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import BaseBackend
def buf_t(x):
    return bytes(x, 'utf8')