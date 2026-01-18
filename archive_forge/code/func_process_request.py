from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def process_request(self, req, resp, *args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(FalconIntegration)
    if integration is None:
        return
    with hub.configure_scope() as scope:
        scope._name = 'falcon'
        scope.add_event_processor(_make_request_event_processor(req, integration))