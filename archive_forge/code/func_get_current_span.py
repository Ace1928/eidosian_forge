import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def get_current_span(hub=None):
    """
    Returns the currently active span if there is one running, otherwise `None`
    """
    if hub is None:
        hub = Hub.current
    current_span = hub.scope.span
    return current_span