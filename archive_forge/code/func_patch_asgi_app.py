from __future__ import absolute_import
import asyncio
import inspect
import threading
from sentry_sdk.hub import _should_send_default_pii, Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def patch_asgi_app():
    old_app = Quart.__call__

    async def sentry_patched_asgi_app(self, scope, receive, send):
        if Hub.current.get_integration(QuartIntegration) is None:
            return await old_app(self, scope, receive, send)
        middleware = SentryAsgiMiddleware(lambda *a, **kw: old_app(self, *a, **kw))
        middleware.__call__ = middleware._run_asgi3
        return await middleware(scope, receive, send)
    Quart.__call__ = sentry_patched_asgi_app