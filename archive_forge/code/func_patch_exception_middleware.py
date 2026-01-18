from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def patch_exception_middleware(middleware_class):
    """
    Capture all exceptions in Starlette app and
    also extract user information.
    """
    old_middleware_init = middleware_class.__init__
    not_yet_patched = '_sentry_middleware_init' not in str(old_middleware_init)
    if not_yet_patched:

        def _sentry_middleware_init(self, *args, **kwargs):
            old_middleware_init(self, *args, **kwargs)
            old_handlers = self._exception_handlers.copy()

            async def _sentry_patched_exception_handler(self, *args, **kwargs):
                exp = args[0]
                is_http_server_error = hasattr(exp, 'status_code') and isinstance(exp.status_code, int) and (exp.status_code >= 500)
                if is_http_server_error:
                    _capture_exception(exp, handled=True)
                old_handler = None
                for cls in type(exp).__mro__:
                    if cls in old_handlers:
                        old_handler = old_handlers[cls]
                        break
                if old_handler is None:
                    return
                if _is_async_callable(old_handler):
                    return await old_handler(self, *args, **kwargs)
                else:
                    return old_handler(self, *args, **kwargs)
            for key in self._exception_handlers.keys():
                self._exception_handlers[key] = _sentry_patched_exception_handler
        middleware_class.__init__ = _sentry_middleware_init
        old_call = middleware_class.__call__

        async def _sentry_exceptionmiddleware_call(self, scope, receive, send):
            _add_user_to_sentry_scope(scope)
            await old_call(self, scope, receive, send)
        middleware_class.__call__ = _sentry_exceptionmiddleware_call