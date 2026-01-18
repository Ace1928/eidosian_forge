import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_write_to_arr_in_tuple(self):
    for func in (global_write_to_arr_in_tuple, global_write_to_arr_in_mixed_tuple):
        jitfunc = njit(func)
        with self.assertRaises(errors.TypingError) as e:
            jitfunc()
        msg = 'Cannot modify readonly array of type:'
        self.assertIn(msg, str(e.exception))