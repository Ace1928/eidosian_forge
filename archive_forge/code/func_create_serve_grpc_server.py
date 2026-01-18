from typing import Sequence
import grpc
from grpc.aio._server import Server
def create_serve_grpc_server(service_handler_factory):
    """Custom function to create Serve's gRPC server.

    This function works similar to `grpc.server()`, but it creates a Serve defined
    gRPC server in order to override the `unary_unary` and `unary_stream` methods

    See: https://grpc.github.io/grpc/python/grpc.html#grpc.server
    """
    return gRPCServer(thread_pool=None, generic_handlers=(), interceptors=(), options=(), maximum_concurrent_rpcs=None, compression=None, service_handler_factory=service_handler_factory)