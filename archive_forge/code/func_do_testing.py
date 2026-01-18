import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def do_testing(self, inputs, dtypes):
    for value, initial_type, expected in inputs:
        for target_type, result in zip(dtypes, expected):
            view = njit(gen_view(initial_type, target_type))
            if not np.isnan(result):
                self.assertEqual(view(value), target_type(result))
                self.assertEqual(view(value), view.py_func(value))
            else:
                self.assertTrue(np.isnan(view(value)))
                self.assertTrue(np.isnan(view.py_func(value)))