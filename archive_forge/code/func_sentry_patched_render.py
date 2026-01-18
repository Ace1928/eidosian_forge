from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import _functools
def sentry_patched_render(self):
    hub = Hub.current
    with hub.start_span(op=OP.VIEW_RESPONSE_RENDER, description='serialize response'):
        return old_render(self)