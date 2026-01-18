from itertools import product
import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.signal._arraytools import axis_slice
def compute_correction_factors_sos(sos, block_sz, dtype):
    n_sections = sos.shape[0]
    correction = cupy.empty((n_sections, 2, block_sz), dtype=dtype)
    corr_kernel = _get_module_func(IIR_SOS_MODULE, 'compute_correction_factors_sos', correction, sos)
    corr_kernel((n_sections,), (2,), (block_sz, sos, correction))
    return correction