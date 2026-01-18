import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def test_ipc_handle_serialization(self):
    for index, foreign in self.variants():
        with self.subTest(index=index, foreign=foreign):
            self.check_ipc_handle_serialization(index, foreign)