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
def getresponse(self, *args, **kwargs):
    span = getattr(self, '_sentrysdk_span', None)
    if span is None:
        return real_getresponse(self, *args, **kwargs)
    rv = real_getresponse(self, *args, **kwargs)
    span.set_http_status(int(rv.status))
    span.set_data('reason', rv.reason)
    span.finish()
    return rv