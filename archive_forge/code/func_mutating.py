import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def mutating(func):
    """Decorator to enforce read-only logic"""

    @functools.wraps(func)
    def wrapped(self, req, *args, **kwargs):
        if req.context.read_only:
            msg = 'Read-only access'
            LOG.debug(msg)
            raise exc.HTTPForbidden(msg, request=req, content_type='text/plain')
        return func(self, req, *args, **kwargs)
    return wrapped