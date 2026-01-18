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
def format_number(value, max_digits, decimal_places):
    """
    Format a number into a string with the requisite number of digits and
    decimal places.
    """
    if value is None:
        return None
    context = decimal.getcontext().copy()
    if max_digits is not None:
        context.prec = max_digits
    if decimal_places is not None:
        value = value.quantize(decimal.Decimal(1).scaleb(-decimal_places), context=context)
    else:
        context.traps[decimal.Rounded] = 1
        value = context.create_decimal(value)
    return '{:f}'.format(value)