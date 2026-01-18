import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
@overload(npy_log2_1p, target='generic')
def ol_npy_log2_1p(x):
    LOG2E = x(_NPY_LOG2E)

    def impl(x):
        return LOG2E * np.log1p(x)
    return impl