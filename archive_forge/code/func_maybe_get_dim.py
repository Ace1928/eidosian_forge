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
def maybe_get_dim(self, name: str) -> Optional[int]:
    """Retrieve the value of a dimension of the given name, or None."""
    return self.get_dim(name) if self.has_dim(name) else None