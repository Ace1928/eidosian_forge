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
def no_4byte_params(f):
    """
    Checks that no 4 byte unicode characters are allowed
    in dicts' keys/values and string's parameters
    """

    def wrapper(*args, **kwargs):

        def _is_match(some_str):
            return isinstance(some_str, str) and REGEX_4BYTE_UNICODE.findall(some_str) != []

        def _check_dict(data_dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    _check_dict(value)
                else:
                    if _is_match(key):
                        msg = _("Property names can't contain 4 byte unicode.")
                        raise exception.Invalid(msg)
                    if _is_match(value):
                        msg = _("%s can't contain 4 byte unicode characters.") % key.title()
                        raise exception.Invalid(msg)
        for data_dict in [arg for arg in args if isinstance(arg, dict)]:
            _check_dict(data_dict)
        for arg in args:
            if _is_match(arg):
                msg = _("Param values can't contain 4 byte unicode.")
                raise exception.Invalid(msg)
        _check_dict(kwargs)
        return f(*args, **kwargs)
    return wrapper