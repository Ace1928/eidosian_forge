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
def maybe_create_breadcrumbs_from_span(hub, span):
    if span.op == OP.DB_REDIS:
        hub.add_breadcrumb(message=span.description, type='redis', category='redis', data=span._tags)
    elif span.op == OP.HTTP_CLIENT:
        hub.add_breadcrumb(type='http', category='httplib', data=span._data)
    elif span.op == 'subprocess':
        hub.add_breadcrumb(type='subprocess', category='subprocess', message=span.description, data=span._data)