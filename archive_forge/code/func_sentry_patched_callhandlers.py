from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
def sentry_patched_callhandlers(self, record):
    ignored_loggers = _IGNORED_LOGGERS
    try:
        return old_callhandlers(self, record)
    finally:
        if ignored_loggers is not None and record.name not in ignored_loggers:
            integration = Hub.current.get_integration(LoggingIntegration)
            if integration is not None:
                integration._handle_record(record)