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
def replace_callbacks(self, forward: Callable, *, init: Optional[Callable]=None) -> None:
    setattr(self, '_func', forward)
    setattr(self, 'init', init)