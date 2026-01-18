import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
@hubmethod
def start_transaction(transaction=None, **kwargs):
    return Hub.current.start_transaction(transaction, **kwargs)