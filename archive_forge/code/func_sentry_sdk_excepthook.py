import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
def sentry_sdk_excepthook(type_, value, traceback):
    hub = Hub.current
    integration = hub.get_integration(ExcepthookIntegration)
    if integration is not None and _should_send(integration.always_run):
        client = hub.client
        with capture_internal_exceptions():
            event, hint = event_from_exception((type_, value, traceback), client_options=client.options, mechanism={'type': 'excepthook', 'handled': False})
            hub.capture_event(event, hint=hint)
    return old_excepthook(type_, value, traceback)