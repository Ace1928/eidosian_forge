from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def observability_deinit() -> None:
    """Clear the observability context, including ObservabilityPlugin and
    ServerCallTracerFactory

    This method have to be called after exit observability context so that
    it's possible to re-initialize again.
    """
    set_plugin(None)
    _cygrpc.clear_server_call_tracer_factory()