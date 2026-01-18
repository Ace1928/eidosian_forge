import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def test_multinomial_2(self):
    """
        Test multinomial(n, pvals)
        """
    cfunc = jit(nopython=True)(numpy_multinomial2)
    n, pvals = (1000, self.pvals)
    res = cfunc(n, pvals)
    self._check_sample(n, pvals, res)
    pvals = list(pvals)
    res = cfunc(n, pvals)
    self._check_sample(n, pvals, res)
    n = 1000000
    pvals = np.array([1, 0, n // 100, 1], dtype=np.float64)
    pvals /= pvals.sum()
    res = cfunc(n, pvals)
    self._check_sample(n, pvals, res)