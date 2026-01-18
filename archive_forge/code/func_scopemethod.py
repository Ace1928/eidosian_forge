import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def scopemethod(f):
    f.__doc__ = '%s\n\n%s' % ('Alias for :py:meth:`sentry_sdk.Scope.%s`' % f.__name__, inspect.getdoc(getattr(Scope, f.__name__)))
    return f