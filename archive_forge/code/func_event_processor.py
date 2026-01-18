import asyncio
import inspect
from copy import deepcopy
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.integrations._asgi_common import (
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction
def event_processor(self, event, hint, asgi_scope):
    request_data = event.get('request', {})
    request_data.update(_get_request_data(asgi_scope))
    event['request'] = deepcopy(request_data)
    already_set = event['transaction'] != _DEFAULT_TRANSACTION_NAME and event['transaction_info'].get('source') in [TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_ROUTE]
    if not already_set:
        name, source = self._get_transaction_name_and_source(self.transaction_style, asgi_scope)
        event['transaction'] = name
        event['transaction_info'] = {'source': source}
        logger.debug("[ASGI] Set transaction name and source in event_processor: '%s' / '%s'", event['transaction'], event['transaction_info']['source'])
    return event