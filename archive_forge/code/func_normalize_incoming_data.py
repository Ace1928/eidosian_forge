import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
def normalize_incoming_data(incoming_data):
    """
    Normalizes incoming data so the keys are all lowercase with dashes instead of underscores and stripped from known prefixes.
    """
    data = {}
    for key, value in incoming_data.items():
        if key.startswith('HTTP_'):
            key = key[5:]
        key = key.replace('_', '-').lower()
        data[key] = value
    return data