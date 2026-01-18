import numpy as np
from numba import cuda
import unittest
from numba.cuda.testing import CUDATestCase
def test_forall_negative_work(self):
    with self.assertRaises(ValueError) as raises:
        foo.forall(-1)
    self.assertIn("Can't create ForAll with negative task count", str(raises.exception))