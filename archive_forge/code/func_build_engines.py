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
def build_engines(entrypoints: EntryPoints) -> dict[str, BackendEntrypoint]:
    backend_entrypoints: dict[str, type[BackendEntrypoint]] = {}
    for backend_name, (module_name, backend) in BACKEND_ENTRYPOINTS.items():
        if module_name is None or module_available(module_name):
            backend_entrypoints[backend_name] = backend
    entrypoints_unique = remove_duplicates(entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(entrypoints_unique)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return {name: backend() for name, backend in backend_entrypoints.items()}