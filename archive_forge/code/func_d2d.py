import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
def d2d(dst, src):
    dst.copy_to_device(src)