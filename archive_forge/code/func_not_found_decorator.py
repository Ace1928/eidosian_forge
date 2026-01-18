import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def not_found_decorator(translated_exc=exceptions.NotFound):
    """Wraps x_wmi: Not Found exceptions as os_win.exceptions.NotFound."""

    def wrapper(func):

        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions.x_wmi as ex:
                if _is_not_found_exc(ex):
                    LOG.debug('x_wmi: Not Found exception raised while running %s', func.__name__)
                    raise translated_exc(message=six.text_type(ex))
                raise
        return inner
    return wrapper