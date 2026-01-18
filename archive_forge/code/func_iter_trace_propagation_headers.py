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
def iter_trace_propagation_headers(self, *args, **kwargs):
    """
        Return HTTP headers which allow propagation of trace data. Data taken
        from the span representing the request, if available, or the current
        span on the scope if not.
        """
    span = kwargs.pop('span', None)
    client = kwargs.pop('client', None)
    propagate_traces = client and client.options['propagate_traces']
    if not propagate_traces:
        return
    span = span or self.span
    if client and has_tracing_enabled(client.options) and (span is not None):
        for header in span.iter_headers():
            yield header
    else:
        for header in self.iter_headers():
            yield header