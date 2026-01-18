from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def tanh_common_test(cc, criterion):
    fastptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32), fastmath=True, cc=cc)
    precptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32), cc=cc)
    criterion.check(self, fastptx, precptx)