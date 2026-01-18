from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
def exception_handler(exc: Exception, scope: 'Scope', _: 'State') -> None:
    hub = Hub.current
    if hub.get_integration(StarliteIntegration) is None:
        return
    user_info: 'Optional[Dict[str, Any]]' = None
    if _should_send_default_pii():
        user_info = retrieve_user_from_scope(scope)
    if user_info and isinstance(user_info, dict):
        with hub.configure_scope() as sentry_scope:
            sentry_scope.set_user(user_info)
    event, hint = event_from_exception(exc, client_options=hub.client.options if hub.client else None, mechanism={'type': StarliteIntegration.identifier, 'handled': False})
    hub.capture_event(event, hint=hint)