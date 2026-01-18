from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
def sentry_patched_wsgi_app(self, environ, start_response):
    if Hub.current.get_integration(FlaskIntegration) is None:
        return old_app(self, environ, start_response)
    return SentryWsgiMiddleware(lambda *a, **kw: old_app(self, *a, **kw))(environ, start_response)