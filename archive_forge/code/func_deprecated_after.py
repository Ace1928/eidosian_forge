import functools
import logging
import os
import pkgutil
import re
import traceback
import warnings
from oslo_utils import strutils
import novaclient
from novaclient import exceptions
from novaclient.i18n import _
def deprecated_after(version):
    decorator = wraps('2.0', version)

    def wrapper(fn):

        @functools.wraps(fn)
        def wrapped(*a, **k):
            decorated = decorator(fn)
            if hasattr(fn, '__module__'):
                mod = fn.__module__
            else:
                mod = a[0].__module__
            warnings.warn('The %s module is deprecated and will be removed.' % mod, DeprecationWarning)
            return decorated(*a, **k)
        return wrapped
    return wrapper