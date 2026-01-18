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
def sentry_patched_wsgi_call(self, environ, start_response):
    hub = Hub.current
    integration = hub.get_integration(PyramidIntegration)
    if integration is None:
        return old_wsgi_call(self, environ, start_response)

    def sentry_patched_inner_wsgi_call(environ, start_response):
        try:
            return old_wsgi_call(self, environ, start_response)
        except Exception:
            einfo = sys.exc_info()
            _capture_exception(einfo)
            reraise(*einfo)
    return SentryWsgiMiddleware(sentry_patched_inner_wsgi_call)(environ, start_response)