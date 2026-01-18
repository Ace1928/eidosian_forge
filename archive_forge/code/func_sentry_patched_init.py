from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions, parse_url, parse_version
def sentry_patched_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    meta = self.meta
    service_id = meta.service_model.service_id.hyphenize()
    meta.events.register('request-created', partial(_sentry_request_created, service_id=service_id))
    meta.events.register('after-call', _sentry_after_call)
    meta.events.register('after-call-error', _sentry_after_call_error)