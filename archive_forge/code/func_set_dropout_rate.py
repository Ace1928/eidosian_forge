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
def set_dropout_rate(model: _ModelT, drop: float, attrs=['dropout_rate']) -> _ModelT:
    """Walk over the model's nodes, setting the dropout rate. You can specify
    one or more attribute names, by default it looks for ["dropout_rate"].
    """
    for node in model.walk():
        for attr in attrs:
            if attr in node.attrs:
                node.attrs[attr] = drop
    return model