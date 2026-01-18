from __future__ import absolute_import
import weakref
from sentry_sdk.consts import OP
from sentry_sdk.api import continue_trace
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def sentry_patched_handle_exception(self, job, *exc_info, **kwargs):
    if job._status == JobStatus.FAILED or job.is_failed:
        _capture_exception(exc_info)
    return old_handle_exception(self, job, *exc_info, **kwargs)