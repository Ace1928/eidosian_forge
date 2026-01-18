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
def serialize_value(self):

    def _hash(x):
        if isinstance(x, str):
            return zlib.crc32(x.encode('utf-8')) & 4294967295
        return int(x)
    return (_hash(value) for value in self.value)