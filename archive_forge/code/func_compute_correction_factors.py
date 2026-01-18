from itertools import product
import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.signal._arraytools import axis_slice
def compute_correction_factors(a, block_sz, dtype):
    k = a.size
    correction = cupy.eye(k, dtype=dtype)
    correction = cupy.c_[correction[::-1], cupy.empty((k, block_sz), dtype=dtype)]
    corr_kernel = _get_module_func(IIR_MODULE, 'compute_correction_factors', correction, a)
    corr_kernel((k,), (1,), (block_sz, k, a, correction))
    return correction