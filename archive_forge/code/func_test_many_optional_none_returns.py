import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_many_optional_none_returns(self):
    """
        Issue #4058
        """

    @njit
    def foo(maybe):
        lx = None
        if maybe:
            lx = 10
        return (1, lx)

    def work():
        tmp = []
        for _ in range(20000):
            maybe = False
            _ = foo(maybe)
    work()