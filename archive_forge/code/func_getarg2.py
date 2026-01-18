import numpy as np
from numba import cuda, float32, int32, void
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from .extensions_usecases import test_struct_model_type
def getarg2(self):
    return self.getarg().reshape(1, 1)