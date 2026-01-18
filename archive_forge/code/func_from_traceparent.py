import uuid
import random
from datetime import datetime, timedelta
import sentry_sdk
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.utils import is_valid_sample_rate, logger, nanosecond_time
from sentry_sdk._compat import datetime_utcnow, utc_from_timestamp, PY2
from sentry_sdk.consts import SPANDATA
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing_utils import (
from sentry_sdk.metrics import LocalAggregator
@classmethod
def from_traceparent(cls, traceparent, **kwargs):
    """
        DEPRECATED: Use :py:meth:`sentry_sdk.tracing.Span.continue_from_headers`.

        Create a ``Transaction`` with the given params, then add in data pulled from
        the given ``sentry-trace`` header value before returning the ``Transaction``.
        """
    logger.warning('Deprecated: Use Transaction.continue_from_headers(headers, **kwargs) instead of from_traceparent(traceparent, **kwargs)')
    if not traceparent:
        return None
    return cls.continue_from_headers({SENTRY_TRACE_HEADER_NAME: traceparent}, **kwargs)