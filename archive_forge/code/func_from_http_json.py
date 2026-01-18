import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def from_http_json(operation, api_request, result_type, **kwargs):
    """Create an operation future using a HTTP/JSON client.

    This interacts with the long-running operations `service`_ (specific
    to a given API) via `HTTP/JSON`_.

    .. _HTTP/JSON: https://cloud.google.com/speech/reference/rest/            v1beta1/operations#Operation

    Args:
        operation (dict): Operation as a dictionary.
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        result_type (:func:`type`): The protobuf result type.
        kwargs: Keyword args passed into the :class:`Operation` constructor.

    Returns:
        ~.api_core.operation.Operation: The operation future to track the given
            operation.
    """
    operation_proto = json_format.ParseDict(operation, operations_pb2.Operation())
    refresh = functools.partial(_refresh_http, api_request, operation_proto.name)
    cancel = functools.partial(_cancel_http, api_request, operation_proto.name)
    return Operation(operation_proto, refresh, cancel, result_type, **kwargs)