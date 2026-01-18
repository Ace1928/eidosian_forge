import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def metrics_noop(func):
    """Convenient decorator that uses `recursion_protection` to
    make a function a noop.
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        with recursion_protection() as in_metrics:
            if not in_metrics:
                return func(*args, **kwargs)
    return new_func