from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def test_expf(self):
    self._test_fast_math_unary(exp, FastMathCriterion(fast_unexpected=['fma.rn.f32 '], prec_expected=['fma.rn.f32 ']))