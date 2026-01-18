import importlib
import inspect
import logging
import os
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter
from types import ModuleType
from typing import (
import ray
import ray._private.worker
from ray._private.inspect_util import (
from ray.runtime_context import get_runtime_context
def inject_current_context() -> Dict[Any, Any]:
    """Inject trace context into otel propagator."""
    context_dict: Dict[Any, Any] = {}
    _opentelemetry.propagate.inject(context_dict)
    return context_dict