import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
@hubmethod
def push_scope(callback=None):
    return Hub.current.push_scope(callback)