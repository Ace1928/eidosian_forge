import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._compat import datetime_utcnow, reraise
from sentry_sdk._types import TYPE_CHECKING
def sentry_handler(aws_event, aws_context, *args, **kwargs):
    if isinstance(aws_event, list):
        request_data = aws_event[0]
        batch_size = len(aws_event)
    else:
        request_data = aws_event
        batch_size = 1
    if not isinstance(request_data, dict):
        request_data = {}
    hub = Hub.current
    integration = hub.get_integration(AwsLambdaIntegration)
    if integration is None:
        return handler(aws_event, aws_context, *args, **kwargs)
    client = hub.client
    configured_time = aws_context.get_remaining_time_in_millis()
    with hub.push_scope() as scope:
        timeout_thread = None
        with capture_internal_exceptions():
            scope.clear_breadcrumbs()
            scope.add_event_processor(_make_request_event_processor(request_data, aws_context, configured_time))
            scope.set_tag('aws_region', aws_context.invoked_function_arn.split(':')[3])
            if batch_size > 1:
                scope.set_tag('batch_request', True)
                scope.set_tag('batch_size', batch_size)
            if integration.timeout_warning and configured_time > TIMEOUT_WARNING_BUFFER:
                waiting_time = (configured_time - TIMEOUT_WARNING_BUFFER) / MILLIS_TO_SECONDS
                timeout_thread = TimeoutThread(waiting_time, configured_time / MILLIS_TO_SECONDS)
                timeout_thread.start()
        headers = request_data.get('headers', {})
        if not isinstance(headers, dict):
            headers = {}
        transaction = continue_trace(headers, op=OP.FUNCTION_AWS, name=aws_context.function_name, source=TRANSACTION_SOURCE_COMPONENT)
        with hub.start_transaction(transaction, custom_sampling_context={'aws_event': aws_event, 'aws_context': aws_context}):
            try:
                return handler(aws_event, aws_context, *args, **kwargs)
            except Exception:
                exc_info = sys.exc_info()
                sentry_event, hint = event_from_exception(exc_info, client_options=client.options, mechanism={'type': 'aws_lambda', 'handled': False})
                hub.capture_event(sentry_event, hint=hint)
                reraise(*exc_info)
            finally:
                if timeout_thread:
                    timeout_thread.stop()