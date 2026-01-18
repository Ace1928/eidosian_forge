import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def exception_translated(exception_type):
    """Wrap API calls to map to correct exception."""

    def _exception_translated(f):

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except exception.PublicIDNotFound as e:
                if exception_type == 'user':
                    raise exception.UserNotFound(user_id=str(e))
                elif exception_type == 'group':
                    raise exception.GroupNotFound(group_id=str(e))
                elif exception_type == 'assertion':
                    raise AssertionError(_('Invalid user / password'))
                else:
                    raise
        return wrapper
    return _exception_translated