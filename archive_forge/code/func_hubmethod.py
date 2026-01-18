import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def hubmethod(f):
    f.__doc__ = '%s\n\n%s' % ('Alias for :py:meth:`sentry_sdk.Hub.%s`' % f.__name__, inspect.getdoc(getattr(Hub, f.__name__)))
    return f