from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import _functools
def patch_views():
    from django.core.handlers.base import BaseHandler
    from django.template.response import SimpleTemplateResponse
    from sentry_sdk.integrations.django import DjangoIntegration
    old_make_view_atomic = BaseHandler.make_view_atomic
    old_render = SimpleTemplateResponse.render

    def sentry_patched_render(self):
        hub = Hub.current
        with hub.start_span(op=OP.VIEW_RESPONSE_RENDER, description='serialize response'):
            return old_render(self)

    @_functools.wraps(old_make_view_atomic)
    def sentry_patched_make_view_atomic(self, *args, **kwargs):
        callback = old_make_view_atomic(self, *args, **kwargs)
        hub = Hub.current
        integration = hub.get_integration(DjangoIntegration)
        if integration is not None and integration.middleware_spans:
            is_async_view = iscoroutinefunction is not None and wrap_async_view is not None and iscoroutinefunction(callback)
            if is_async_view:
                sentry_wrapped_callback = wrap_async_view(hub, callback)
            else:
                sentry_wrapped_callback = _wrap_sync_view(hub, callback)
        else:
            sentry_wrapped_callback = callback
        return sentry_wrapped_callback
    SimpleTemplateResponse.render = sentry_patched_render
    BaseHandler.make_view_atomic = sentry_patched_make_view_atomic