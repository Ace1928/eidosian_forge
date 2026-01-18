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
def gauge(key, value, unit='none', tags=None, timestamp=None, stacklevel=0):
    """Emits a gauge."""
    aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(key, tags)
    if aggregator is not None:
        aggregator.add('g', key, value, unit, tags, timestamp, local_aggregator, stacklevel)