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
def extract_sentrytrace_data(header):
    """
    Given a `sentry-trace` header string, return a dictionary of data.
    """
    if not header:
        return None
    if header.startswith('00-') and header.endswith('-00'):
        header = header[3:-3]
    match = SENTRY_TRACE_REGEX.match(header)
    if not match:
        return None
    trace_id, parent_span_id, sampled_str = match.groups()
    parent_sampled = None
    if trace_id:
        trace_id = '{:032x}'.format(int(trace_id, 16))
    if parent_span_id:
        parent_span_id = '{:016x}'.format(int(parent_span_id, 16))
    if sampled_str:
        parent_sampled = sampled_str != '0'
    return {'trace_id': trace_id, 'parent_span_id': parent_span_id, 'parent_sampled': parent_sampled}