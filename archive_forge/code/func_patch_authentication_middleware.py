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
def patch_authentication_middleware(middleware_class):
    """
    Add user information to Sentry scope.
    """
    old_call = middleware_class.__call__
    not_yet_patched = '_sentry_authenticationmiddleware_call' not in str(old_call)
    if not_yet_patched:

        async def _sentry_authenticationmiddleware_call(self, scope, receive, send):
            await old_call(self, scope, receive, send)
            _add_user_to_sentry_scope(scope)
        middleware_class.__call__ = _sentry_authenticationmiddleware_call