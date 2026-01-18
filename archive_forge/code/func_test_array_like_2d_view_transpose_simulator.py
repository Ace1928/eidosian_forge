import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
@skip_unless_cudasim('Numba and NumPy stride semantics differ for transpose')
def test_array_like_2d_view_transpose_simulator(self):
    shape = (10, 12)
    view = np.zeros(shape)[::2, ::2].T
    d_view = cuda.device_array(shape)[::2, ::2].T
    for like_func in ARRAY_LIKE_FUNCTIONS:
        with self.subTest(like_func=like_func):
            np_like = np.zeros_like(view)
            nb_like = like_func(d_view)
            self.assertEqual(d_view.shape, nb_like.shape)
            self.assertEqual(d_view.dtype, nb_like.dtype)
            self.assertEqual(np_like.strides, nb_like.strides)
            self.assertEqual(np_like.flags['C_CONTIGUOUS'], nb_like.flags['C_CONTIGUOUS'])
            self.assertEqual(np_like.flags['F_CONTIGUOUS'], nb_like.flags['F_CONTIGUOUS'])