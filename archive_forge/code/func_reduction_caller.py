import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
from numba.tests.support import skip_unless_cffi
@cuda.jit(link=[functions_cu])
def reduction_caller(result, array):
    array_ptr = ffi.from_buffer(array)
    result[()] = sum_reduce(array_ptr, len(array))