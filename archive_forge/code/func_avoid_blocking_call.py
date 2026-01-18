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
def avoid_blocking_call(f, *args, **kwargs):
    """Ensures that the invoked method will not block other greenthreads.

    Performs the call in a different thread using tpool.execute when called
    from a greenthread.
    """
    if eventlet.getcurrent().parent:
        return tpool.execute(f, *args, **kwargs)
    else:
        return f(*args, **kwargs)