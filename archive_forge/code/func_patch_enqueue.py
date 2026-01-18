from __future__ import absolute_import
import sys
from datetime import datetime
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.api import continue_trace, get_baggage, get_traceparent
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
def patch_enqueue():
    old_enqueue = Huey.enqueue

    def _sentry_enqueue(self, task):
        hub = Hub.current
        if hub.get_integration(HueyIntegration) is None:
            return old_enqueue(self, task)
        with hub.start_span(op=OP.QUEUE_SUBMIT_HUEY, description=task.name):
            if not isinstance(task, PeriodicTask):
                task.kwargs['sentry_headers'] = {BAGGAGE_HEADER_NAME: get_baggage(), SENTRY_TRACE_HEADER_NAME: get_traceparent()}
            return old_enqueue(self, task)
    Huey.enqueue = _sentry_enqueue