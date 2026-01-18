import weakref
import contextlib
from inspect import iscoroutinefunction
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
def sentry_log_exception(self, ty, value, tb, *args, **kwargs):
    _capture_exception(ty, value, tb)
    return old_log_exception(self, ty, value, tb, *args, **kwargs)