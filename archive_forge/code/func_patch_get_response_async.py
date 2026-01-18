import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def patch_get_response_async(cls, _before_get_response):
    old_get_response_async = cls.get_response_async

    async def sentry_patched_get_response_async(self, request):
        _before_get_response(request)
        return await old_get_response_async(self, request)
    cls.get_response_async = sentry_patched_get_response_async