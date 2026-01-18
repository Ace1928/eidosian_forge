import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def stream_unary_rpc_method_handler(behavior, request_deserializer=None, response_serializer=None):
    """Creates an RpcMethodHandler for a stream-unary RPC method.

    Args:
      behavior: The implementation of an RPC that accepts an iterator of
        request values and returns a single response value.
      request_deserializer: An optional :term:`deserializer` for request deserialization.
      response_serializer: An optional :term:`serializer` for response serialization.

    Returns:
      An RpcMethodHandler object that is typically used by grpc.Server.
    """
    from grpc import _utilities
    return _utilities.RpcMethodHandler(True, False, request_deserializer, response_serializer, None, None, behavior, None)