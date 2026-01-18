from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def from_grpc_status(status_code, message, **kwargs):
    """Create a :class:`GoogleAPICallError` from a :class:`grpc.StatusCode`.

    Args:
        status_code (Union[grpc.StatusCode, int]): The gRPC status code.
        message (str): The exception message.
        kwargs: Additional arguments passed to the :class:`GoogleAPICallError`
            constructor.

    Returns:
        GoogleAPICallError: An instance of the appropriate subclass of
            :class:`GoogleAPICallError`.
    """
    if isinstance(status_code, int):
        status_code = _INT_TO_GRPC_CODE.get(status_code, status_code)
    error_class = exception_class_for_grpc_status(status_code)
    error = error_class(message, **kwargs)
    if error.grpc_status_code is None:
        error.grpc_status_code = status_code
    return error