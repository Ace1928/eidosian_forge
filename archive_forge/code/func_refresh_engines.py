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
def refresh_engines() -> None:
    """Refreshes the backend engines based on installed packages."""
    list_engines.cache_clear()