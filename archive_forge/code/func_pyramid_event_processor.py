from __future__ import absolute_import
import os
import sys
import weakref
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._compat import reraise, iteritems
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk._types import TYPE_CHECKING
def pyramid_event_processor(event, hint):
    request = weak_request()
    if request is None:
        return event
    with capture_internal_exceptions():
        PyramidRequestExtractor(request).extract_into_event(event)
    if _should_send_default_pii():
        with capture_internal_exceptions():
            user_info = event.setdefault('user', {})
            user_info.setdefault('id', authenticated_userid(request))
    return event