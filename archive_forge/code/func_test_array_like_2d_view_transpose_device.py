import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
@skip_on_cudasim('Numba and NumPy stride semantics differ for transpose')
def test_array_like_2d_view_transpose_device(self):
    shape = (10, 12)
    d_view = cuda.device_array(shape)[::2, ::2].T
    for like_func in ARRAY_LIKE_FUNCTIONS:
        with self.subTest(like_func=like_func):
            like = like_func(d_view)
            self.assertEqual(d_view.shape, like.shape)
            self.assertEqual(d_view.dtype, like.dtype)
            self.assertEqual((40, 8), like.strides)
            self.assertTrue(like.flags['C_CONTIGUOUS'])
            self.assertFalse(like.flags['F_CONTIGUOUS'])