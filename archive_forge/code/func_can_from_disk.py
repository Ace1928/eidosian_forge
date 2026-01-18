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
def can_from_disk(self, path: Union[Path, str], *, strict: bool=True) -> bool:
    """Check whether serialized data on disk is compatible with the model.
        If 'strict', the function returns False if the model has an attribute
        already loaded that would be changed.
        """
    path = Path(path) if isinstance(path, str) else path
    if path.is_dir() or not path.exists():
        return False
    with path.open('rb') as file_:
        bytes_data = file_.read()
    return self.can_from_bytes(bytes_data, strict=strict)