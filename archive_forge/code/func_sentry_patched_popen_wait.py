import os
import subprocess
import sys
import platform
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing_utils import EnvironHeaders, should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def sentry_patched_popen_wait(self, *a, **kw):
    hub = Hub.current
    if hub.get_integration(StdlibIntegration) is None:
        return old_popen_wait(self, *a, **kw)
    with hub.start_span(op=OP.SUBPROCESS_WAIT) as span:
        span.set_tag('subprocess.pid', self.pid)
        return old_popen_wait(self, *a, **kw)