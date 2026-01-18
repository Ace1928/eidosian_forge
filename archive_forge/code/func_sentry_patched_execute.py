from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def sentry_patched_execute(self, *args, **kwargs):
    hub = Hub.current
    if hub.get_integration(RedisIntegration) is None:
        return old_execute(self, *args, **kwargs)
    with hub.start_span(op=OP.DB_REDIS, description='redis.pipeline.execute') as span:
        with capture_internal_exceptions():
            set_db_data_fn(span, self)
            _set_pipeline_data(span, is_cluster, get_command_args_fn, False if is_cluster else self.transaction, self.command_stack)
        return old_execute(self, *args, **kwargs)