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
@metrics_noop
def need_code_loation(self, ty, key, unit, timestamp):
    if self._enable_code_locations:
        return False
    meta_key = (ty, key, unit)
    start_of_day = utc_from_timestamp(timestamp).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    start_of_day = int(to_timestamp(start_of_day))
    return (start_of_day, meta_key) not in self._seen_locations