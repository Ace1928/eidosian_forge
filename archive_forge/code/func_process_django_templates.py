from __future__ import absolute_import
import inspect
import sys
import threading
import weakref
from importlib import import_module
from sentry_sdk._compat import string_types, text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.db.explain_plan.django import attach_explain_plan_to_span
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.serializer import add_global_repr_processor
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_URL
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.django.transactions import LEGACY_RESOLVER
from sentry_sdk.integrations.django.templates import (
from sentry_sdk.integrations.django.middleware import patch_django_middlewares
from sentry_sdk.integrations.django.signals_handlers import patch_signals
from sentry_sdk.integrations.django.views import patch_views
@add_global_event_processor
def process_django_templates(event, hint):
    if hint is None:
        return event
    exc_info = hint.get('exc_info', None)
    if exc_info is None:
        return event
    exception = event.get('exception', None)
    if exception is None:
        return event
    values = exception.get('values', None)
    if values is None:
        return event
    for exception, (_, exc_value, _) in zip(reversed(values), walk_exception_chain(exc_info)):
        frame = get_template_frame_from_exception(exc_value)
        if frame is not None:
            frames = exception.get('stacktrace', {}).get('frames', [])
            for i in reversed(range(len(frames))):
                f = frames[i]
                if f.get('function') in ('Parser.parse', 'parse', 'render') and f.get('module') == 'django.template.base':
                    i += 1
                    break
            else:
                i = len(frames)
            frames.insert(i, frame)
    return event