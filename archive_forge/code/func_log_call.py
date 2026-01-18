import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def log_call(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        LOG.info(_LI('Calling %(funcname)s: args=%(args)s, kwargs=%(kwargs)s'), {'funcname': func.__name__, 'args': args, 'kwargs': kwargs})
        output = func(*args, **kwargs)
        LOG.info(_LI('Returning %(funcname)s: %(output)s'), {'funcname': func.__name__, 'output': output})
        return output
    return wrapped