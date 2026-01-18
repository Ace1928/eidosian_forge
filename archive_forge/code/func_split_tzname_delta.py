import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
def split_tzname_delta(tzname):
    """
    Split a time zone name into a 3-tuple of (name, sign, offset).
    """
    for sign in ['+', '-']:
        if sign in tzname:
            name, offset = tzname.rsplit(sign, 1)
            if offset and parse_time(offset):
                return (name, sign, offset)
    return (tzname, None, None)