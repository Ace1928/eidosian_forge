from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def sentry_patched_execute_command(self, name, *args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(RedisIntegration)
    if integration is None:
        return old_execute_command(self, name, *args, **kwargs)
    description = _get_span_description(name, *args)
    data_should_be_truncated = integration.max_data_size and len(description) > integration.max_data_size
    if data_should_be_truncated:
        description = description[:integration.max_data_size - len('...')] + '...'
    with hub.start_span(op=OP.DB_REDIS, description=description) as span:
        set_db_data_fn(span, self)
        _set_client_data(span, is_cluster, name, *args)
        return old_execute_command(self, name, *args, **kwargs)