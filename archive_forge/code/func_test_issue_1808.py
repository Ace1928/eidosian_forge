import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_issue_1808(self):
    """
        Incorrect return data model
        """
    magic = 3735928559

    @njit
    def generator():
        yield magic

    @njit
    def get_generator():
        return generator()

    @njit
    def main():
        out = 0
        for x in get_generator():
            out += x
        return out
    self.assertEqual(main(), magic)