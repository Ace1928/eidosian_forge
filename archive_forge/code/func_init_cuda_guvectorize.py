from numba.np.ufunc.decorators import Vectorize, GUVectorize, vectorize, guvectorize
from numba.np.ufunc._internal import PyUFunc_None, PyUFunc_Zero, PyUFunc_One
from numba.np.ufunc import _internal, array_exprs
from numba.np.ufunc.parallel import (threading_layer, get_num_threads,
def init_cuda_guvectorize():
    from numba.cuda.vectorizers import CUDAGUFuncVectorize
    return CUDAGUFuncVectorize