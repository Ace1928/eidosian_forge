import random
import numpy as np
from numba import njit
from numba.core import types
import unittest
def test_multi3(self):

    @njit('(int64,)')
    def func(x):
        res = 0
        for i in range(x):
            res += i
        return res
    x_cases = [-1, 0, 1, 3, 4, 8, 4294967295 - 1, 4294967295, 4294967295 + 1, 81985529216486895, -81985529216486895]
    for _ in range(500):
        x_cases.append(random.randint(0, 4294967295))

    def expected(x):
        if x <= 0:
            return 0
        return x * (x - 1) // 2 & 2 ** 64 - 1
    for x in x_cases:
        self.assertEqual(expected(x), func(x))