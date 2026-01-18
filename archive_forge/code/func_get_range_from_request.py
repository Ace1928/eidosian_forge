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
def get_range_from_request(self, image_size):
    """Return the `Range` in a request."""
    range_str = self.headers.get('Range')
    if range_str is not None:
        if ',' in range_str:
            msg = 'Requests with multiple ranges are not supported in Glance. You may make multiple single-range requests instead.'
            raise webob.exc.HTTPBadRequest(explanation=msg)
        range_ = webob.byterange.Range.parse(range_str)
        if range_ is None:
            msg = 'Invalid Range header.'
            raise webob.exc.HTTPRequestRangeNotSatisfiable(msg)
        if range_.start >= image_size:
            msg = 'Invalid start position in Range header. Start position MUST be in the inclusive range [0, %s].' % (image_size - 1)
            raise webob.exc.HTTPRequestRangeNotSatisfiable(msg)
        return range_
    c_range_str = self.headers.get('Content-Range')
    if c_range_str is not None:
        content_range = webob.byterange.ContentRange.parse(c_range_str)
        if content_range is None:
            msg = 'Invalid Content-Range header.'
            raise webob.exc.HTTPRequestRangeNotSatisfiable(msg)
        if content_range.length is None and content_range.stop > image_size:
            msg = 'Invalid stop position in Content-Range header. The stop position MUST be in the inclusive range [0, %s].' % (image_size - 1)
            raise webob.exc.HTTPRequestRangeNotSatisfiable(msg)
        if content_range.start >= image_size:
            msg = 'Invalid start position in Content-Range header. Start position MUST be in the inclusive range [0, %s].' % (image_size - 1)
            raise webob.exc.HTTPRequestRangeNotSatisfiable(msg)
        return content_range