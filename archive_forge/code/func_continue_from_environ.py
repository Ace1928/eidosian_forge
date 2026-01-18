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
def continue_from_environ(cls, environ, **kwargs):
    """
        Create a Transaction with the given params, then add in data pulled from
        the ``sentry-trace`` and ``baggage`` headers from the environ (if any)
        before returning the Transaction.

        This is different from :py:meth:`~sentry_sdk.tracing.Span.continue_from_headers`
        in that it assumes header names in the form ``HTTP_HEADER_NAME`` -
        such as you would get from a WSGI/ASGI environ -
        rather than the form ``header-name``.

        :param environ: The ASGI/WSGI environ to pull information from.
        """
    if cls is Span:
        logger.warning('Deprecated: use Transaction.continue_from_environ instead of Span.continue_from_environ.')
    return Transaction.continue_from_headers(EnvironHeaders(environ), **kwargs)