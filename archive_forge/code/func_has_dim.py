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
def has_dim(self, name: str) -> Optional[bool]:
    """Check whether the model has a dimension of a given name. If the
        dimension is registered but the value is unset, returns None.
        """
    if name not in self._dims:
        return False
    elif self._dims[name] is not None:
        return True
    else:
        return None