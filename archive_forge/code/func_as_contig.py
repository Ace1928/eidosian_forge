import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def as_contig(self, data: ArrayT, dtype: Optional[DTypes]=None) -> ArrayT:
    """Allow the backend to make a contiguous copy of an array.
        Implementations of `Ops` do not have to make a copy or make it
        contiguous if that would not improve efficiency for the execution engine.
        """
    if data.flags['C_CONTIGUOUS'] and dtype in (None, data.dtype):
        return data
    kwargs = {'dtype': dtype} if dtype is not None else {}
    return self.xp.ascontiguousarray(data, **kwargs)