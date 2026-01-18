import numpy as np
from numba import cuda, types
from numba.cuda.testing import (skip_on_cudasim, test_data_dir, unittest,
from numba.tests.support import skip_unless_cffi
@cuda.jit(link=[link])
def mutate_array(x):
    x_ptr = ffi.from_buffer(x)
    array_mutator(x_ptr)