import sys
import weakref
from sentry_sdk.api import continue_trace
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def get_aiohttp_request_data(hub, request):
    bytes_body = request._read_bytes
    if bytes_body is not None:
        if not request_body_within_bounds(hub.client, len(bytes_body)):
            return AnnotatedValue.removed_because_over_size_limit()
        encoding = request.charset or 'utf-8'
        return bytes_body.decode(encoding, 'replace')
    if request.can_read_body:
        return BODY_NOT_READ_MESSAGE
    return None