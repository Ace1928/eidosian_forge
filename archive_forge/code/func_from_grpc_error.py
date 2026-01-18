from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def from_grpc_error(rpc_exc):
    """Create a :class:`GoogleAPICallError` from a :class:`grpc.RpcError`.

    Args:
        rpc_exc (grpc.RpcError): The gRPC error.

    Returns:
        GoogleAPICallError: An instance of the appropriate subclass of
            :class:`GoogleAPICallError`.
    """
    if grpc is not None and isinstance(rpc_exc, grpc.Call) or _is_informative_grpc_error(rpc_exc):
        details, err_info = _parse_grpc_error_details(rpc_exc)
        return from_grpc_status(rpc_exc.code(), rpc_exc.details(), errors=(rpc_exc,), details=details, response=rpc_exc, error_info=err_info)
    else:
        return GoogleAPICallError(str(rpc_exc), errors=(rpc_exc,), response=rpc_exc)