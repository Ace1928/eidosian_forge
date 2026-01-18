import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
@property
def out_dtype(self):
    """Return `out_dtype` from arraywriter"""
    return self._out_dtype