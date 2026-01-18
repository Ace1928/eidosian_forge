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
def dynamic_sampling_context(self):
    header = {}
    for key, item in iteritems(self.sentry_items):
        header[key] = item
    return header