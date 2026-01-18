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
def intercept_channel(channel, *interceptors):
    """Intercepts a channel through a set of interceptors.

    Args:
      channel: A Channel.
      interceptors: Zero or more objects of type
        UnaryUnaryClientInterceptor,
        UnaryStreamClientInterceptor,
        StreamUnaryClientInterceptor, or
        StreamStreamClientInterceptor.
        Interceptors are given control in the order they are listed.

    Returns:
      A Channel that intercepts each invocation via the provided interceptors.

    Raises:
      TypeError: If interceptor does not derive from any of
        UnaryUnaryClientInterceptor,
        UnaryStreamClientInterceptor,
        StreamUnaryClientInterceptor, or
        StreamStreamClientInterceptor.
    """
    from grpc import _interceptor
    return _interceptor.intercept_channel(channel, *interceptors)