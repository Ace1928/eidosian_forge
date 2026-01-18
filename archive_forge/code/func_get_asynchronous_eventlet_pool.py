import abc
import errno
import functools
import os
import re
import signal
import struct
import subprocess
import sys
import time
from eventlet.green import socket
import eventlet.greenio
import eventlet.wsgi
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from osprofiler import opts as profiler_opts
import routes.middleware
import webob.dec
import webob.exc
from webob import multidict
from glance.common import config
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
import glance.db
from glance import housekeeping
from glance import i18n
from glance.i18n import _, _LE, _LI, _LW
from glance import sqlite_migration
def get_asynchronous_eventlet_pool(size=1000):
    """Return eventlet pool to caller.

    Also store pools created in global list, to wait on
    it after getting signal for graceful shutdown.

    :param size: eventlet pool size
    :returns: eventlet pool
    """
    global ASYNC_EVENTLET_THREAD_POOL_LIST
    pool = eventlet.GreenPool(size=size)
    ASYNC_EVENTLET_THREAD_POOL_LIST.append(pool)
    return pool