import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
@unittest.skipIf(not cuda.is_available(), 'NO CUDA')
def test_cuda_submodules(self):
    self.check_listing_prefix('numba.cuda.tests.cudadrv')
    self.check_listing_prefix('numba.cuda.tests.cudapy')
    self.check_listing_prefix('numba.cuda.tests.nocuda')
    self.check_listing_prefix('numba.cuda.tests.cudasim')