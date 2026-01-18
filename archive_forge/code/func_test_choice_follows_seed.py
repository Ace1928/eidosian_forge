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
def test_choice_follows_seed(self):

    @jit(nopython=True)
    def numba_rands(n_to_return, choice_array):
        np.random.seed(1337)
        out = np.empty((n_to_return, 2), np.int32)
        for i in range(n_to_return):
            out[i] = np.random.choice(choice_array, 2, False)
        return out
    choice_array = np.random.randint(300, size=1000).astype(np.int32)
    tmp_np = choice_array.copy()
    expected = numba_rands.py_func(5, tmp_np)
    tmp_nb = choice_array.copy()
    got = numba_rands(5, tmp_nb)
    np.testing.assert_allclose(expected, got)
    np.testing.assert_allclose(choice_array, tmp_np)
    np.testing.assert_allclose(choice_array, tmp_nb)