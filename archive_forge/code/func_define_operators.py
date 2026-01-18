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
@classmethod
@contextlib.contextmanager
def define_operators(cls, operators: Dict[str, Callable]):
    """Bind arbitrary binary functions to Python operators, for use in any
        `Model` instance. Can (and should) be used as a contextmanager.

        EXAMPLE:
            with Model.define_operators({">>": chain}):
                model = Relu(512) >> Relu(512) >> Softmax()
        """
    token = cls._context_operators.set(dict(operators))
    yield
    cls._context_operators.reset(token)