import contextlib
import copy
import functools
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import (
import srsly
from .backends import CupyOps, NumpyOps, Ops, ParamServer, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .types import FloatsXd
from .util import (
def set_dim(self, name: str, value: int, *, force: bool=False) -> None:
    """Set a value for a dimension."""
    if name not in self._dims:
        raise KeyError(f"Cannot set unknown dimension '{name}' for model '{self.name}'.")
    old_value = self._dims[name]
    has_params = any((bool(y) for x, y in self._has_params.items()))
    invalid_change = (old_value is not None and old_value != value) and (not force or (force and has_params))
    if invalid_change:
        err = f"Attempt to change dimension '{name}' for model '{self.name}' from {old_value} to {value}"
        raise ValueError(err)
    self._dims[name] = value