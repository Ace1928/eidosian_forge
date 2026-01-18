from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
def injection_wrapper(self: 'Starlite', *args: 'Any', **kwargs: 'Any') -> None:
    after_exception = kwargs.pop('after_exception', [])
    kwargs.update(after_exception=[exception_handler, *(after_exception if isinstance(after_exception, list) else [after_exception])])
    SentryStarliteASGIMiddleware.__call__ = SentryStarliteASGIMiddleware._run_asgi3
    middleware = kwargs.pop('middleware', None) or []
    kwargs['middleware'] = [SentryStarliteASGIMiddleware, *middleware]
    old__init__(self, *args, **kwargs)