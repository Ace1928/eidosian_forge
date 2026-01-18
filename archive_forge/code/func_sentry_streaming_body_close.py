from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions, parse_url, parse_version
def sentry_streaming_body_close(*args, **kwargs):
    streaming_span.finish()
    orig_close(*args, **kwargs)