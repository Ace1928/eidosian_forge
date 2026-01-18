from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def set_plugin(observability_plugin: Optional[ObservabilityPlugin]) -> None:
    """Save ObservabilityPlugin to _observability module.

    Args:
    observability_plugin: The ObservabilityPlugin to save.

    Raises:
      ValueError: If an ObservabilityPlugin was already registered at the
      time of calling this method.
    """
    global _OBSERVABILITY_PLUGIN
    with _plugin_lock:
        if observability_plugin and _OBSERVABILITY_PLUGIN:
            raise ValueError('observability_plugin was already set!')
        _OBSERVABILITY_PLUGIN = observability_plugin