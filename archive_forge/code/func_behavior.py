from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_CUSTOM
def behavior(request, context):
    hub = Hub(Hub.current)
    name = self._find_method_name(context)
    if name:
        metadata = dict(context.invocation_metadata())
        transaction = Transaction.continue_from_headers(metadata, op=OP.GRPC_SERVER, name=name, source=TRANSACTION_SOURCE_CUSTOM)
        with hub.start_transaction(transaction=transaction):
            try:
                return handler.unary_unary(request, context)
            except BaseException as e:
                raise e
    else:
        return handler.unary_unary(request, context)