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
def get_dim(self, name: str) -> int:
    """Retrieve the value of a dimension of the given name."""
    if name not in self._dims:
        raise KeyError(f"Cannot get dimension '{name}' for model '{self.name}'")
    value = self._dims[name]
    if value is None:
        err = f"Cannot get dimension '{name}' for model '{self.name}': value unset"
        raise ValueError(err)
    else:
        return value