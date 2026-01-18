from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
@property
def mT(self) -> Array:
    from .linalg import matrix_transpose
    return matrix_transpose(self)