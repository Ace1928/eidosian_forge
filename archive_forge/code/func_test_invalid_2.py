import numpy as np
from numba import cuda, float32, int32, void
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from .extensions_usecases import test_struct_model_type
@skip_on_cudasim("Can't check for constants in simulator")
def test_invalid_2(self):
    with self.assertRaises(TypingError) as raises:
        cuda.jit((float32[:, :],))(udt_invalid_2)
    self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
    self.assertIn('found for signature:\n \n >>> array(shape=Tuple(Literal[int](1), array(float32, 1d, A)), dtype=class(float32))', str(raises.exception))