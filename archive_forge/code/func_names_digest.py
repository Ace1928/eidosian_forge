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
def names_digest(*args, length):
    """
    Generate a 32-bit digest of a set of arguments that can be used to shorten
    identifying names.
    """
    h = md5(usedforsecurity=False)
    for arg in args:
        h.update(arg.encode())
    return h.hexdigest()[:length]