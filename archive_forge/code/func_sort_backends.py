from __future__ import annotations
import functools
import inspect
import itertools
import sys
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.utils import module_available
def sort_backends(backend_entrypoints: dict[str, type[BackendEntrypoint]]) -> dict[str, type[BackendEntrypoint]]:
    ordered_backends_entrypoints = {}
    for be_name in STANDARD_BACKENDS_ORDER:
        if backend_entrypoints.get(be_name, None) is not None:
            ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
    ordered_backends_entrypoints.update({name: backend_entrypoints[name] for name in sorted(backend_entrypoints)})
    return ordered_backends_entrypoints