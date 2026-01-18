from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def iter_headers(self):
    """
        Creates a generator which returns the `sentry-trace` and `baggage` headers from the Propagation Context.
        """
    if self._propagation_context is not None:
        traceparent = self.get_traceparent()
        if traceparent is not None:
            yield (SENTRY_TRACE_HEADER_NAME, traceparent)
        dsc = self.get_dynamic_sampling_context()
        if dsc is not None:
            baggage = Baggage(dsc).serialize()
            yield (BAGGAGE_HEADER_NAME, baggage)