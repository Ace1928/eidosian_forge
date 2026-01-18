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
def stash_conf_values():
    """
    Make a copy of some of the current global CONF's settings.
    Allows determining if any of these values have changed
    when the config is reloaded.
    """
    conf = {'bind_host': CONF.bind_host, 'bind_port': CONF.bind_port, 'backlog': CONF.backlog}
    return conf