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
def method_handlers_generic_handler(service, method_handlers):
    """Creates a GenericRpcHandler from RpcMethodHandlers.

    Args:
      service: The name of the service that is implemented by the
        method_handlers.
      method_handlers: A dictionary that maps method names to corresponding
        RpcMethodHandler.

    Returns:
      A GenericRpcHandler. This is typically added to the grpc.Server object
      with add_generic_rpc_handlers() before starting the server.
    """
    from grpc import _utilities
    return _utilities.DictionaryGenericHandler(service, method_handlers)