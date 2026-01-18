import sys
from sentry_sdk._compat import PY2, reraise
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._werkzeug import get_host, _get_headers
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import _filter_headers
def get_request_url(environ, use_x_forwarded_for=False):
    """Return the absolute URL without query string for the given WSGI
    environment."""
    return '%s://%s/%s' % (environ.get('wsgi.url_scheme'), get_host(environ, use_x_forwarded_for), wsgi_decoding_dance(environ.get('PATH_INFO') or '').lstrip('/'))