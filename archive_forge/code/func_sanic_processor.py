import sys
import weakref
from inspect import isawaitable
from sentry_sdk import continue_trace
from sentry_sdk._compat import urlparse, reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_URL
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor, _filter_headers
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def sanic_processor(event, hint):
    try:
        if hint and issubclass(hint['exc_info'][0], SanicException):
            return None
    except KeyError:
        pass
    request = weak_request()
    if request is None:
        return event
    with capture_internal_exceptions():
        extractor = SanicRequestExtractor(request)
        extractor.extract_into_event(event)
        request_info = event['request']
        urlparts = urlparse.urlsplit(request.url)
        request_info['url'] = '%s://%s%s' % (urlparts.scheme, urlparts.netloc, urlparts.path)
        request_info['query_string'] = urlparts.query
        request_info['method'] = request.method
        request_info['env'] = {'REMOTE_ADDR': request.remote_addr}
        request_info['headers'] = _filter_headers(dict(request.headers))
    return event