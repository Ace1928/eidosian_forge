from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@abc.abstractmethod
def create_client_call_tracer(self, method_name: bytes, target: bytes) -> ClientCallTracerCapsule:
    """Creates a ClientCallTracerCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called after a call was created, the ClientCallTracer created
        by this method will be saved to call context.

        The ClientCallTracer is an object which implements `grpc_core::ClientCallTracer`
        interface and wrapped in a PyCapsule using `client_call_tracer` as name.

        Args:
        method_name: The method name of the call in byte format.
        target: The channel target of the call in byte format.

        Returns:
        A PyCapsule which stores a ClientCallTracer object.
        """
    raise NotImplementedError()