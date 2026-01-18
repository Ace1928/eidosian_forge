from time import time
from opentelemetry.context import get_value  # type: ignore
from opentelemetry.sdk.trace import SpanProcessor  # type: ignore
from opentelemetry.semconv.trace import SpanAttributes  # type: ignore
from opentelemetry.trace import (  # type: ignore
from opentelemetry.trace.span import (  # type: ignore
from sentry_sdk._compat import utc_from_timestamp
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.hub import Hub
from sentry_sdk.integrations.opentelemetry.consts import (
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing import Transaction, Span as SentrySpan
from sentry_sdk.utils import Dsn
from sentry_sdk._types import TYPE_CHECKING
from urllib3.util import parse_url as urlparse
def on_end(self, otel_span):
    hub = Hub.current
    if not hub:
        return
    if hub.client and hub.client.options['instrumenter'] != INSTRUMENTER.OTEL:
        return
    span_context = otel_span.get_span_context()
    if not span_context.is_valid:
        return
    span_id = format_span_id(span_context.span_id)
    sentry_span = self.otel_span_map.pop(span_id, None)
    if not sentry_span:
        return
    sentry_span.op = otel_span.name
    self._update_span_with_otel_status(sentry_span, otel_span)
    if isinstance(sentry_span, Transaction):
        sentry_span.name = otel_span.name
        sentry_span.set_context(OPEN_TELEMETRY_CONTEXT, self._get_otel_context(otel_span))
        self._update_transaction_with_otel_data(sentry_span, otel_span)
    else:
        self._update_span_with_otel_data(sentry_span, otel_span)
    sentry_span.finish(end_timestamp=utc_from_timestamp(otel_span.end_time / 1000000000.0))
    span_start_in_minutes = int(otel_span.start_time / 1000000000.0 / 60)
    self.open_spans.setdefault(span_start_in_minutes, set()).discard(span_id)
    self._prune_old_spans()