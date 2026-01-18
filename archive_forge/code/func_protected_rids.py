import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def protected_rids() -> Tuple[Tuple[int, int], ...]:
    """Sequence of R IDs protected from collection by rpy2."""
    keys = tuple(_R_PRESERVED.keys())
    res = []
    for k in keys:
        v = _R_PRESERVED.get(k)
        if v:
            res.append((get_rid(k), v))
    return tuple(res)