from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
@classmethod
def from_int(cls, retries, redirect=True, default=None):
    """Backwards-compatibility for the old retries format."""
    if retries is None:
        retries = default if default is not None else cls.DEFAULT
    if isinstance(retries, Retry):
        return retries
    redirect = bool(redirect) and None
    new_retries = cls(retries, redirect=redirect)
    log.debug('Converted retries value: %r -> %r', retries, new_retries)
    return new_retries