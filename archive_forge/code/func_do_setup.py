import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def do_setup(deterministic=True):
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    _old_python_random_state = random.getstate()
    _old_numpy_random_state = numpy.random.get_state()
    _old_cupy_random_states = cupy.random._generator._random_states
    cupy.random.reset_states()
    assert cupy.random._generator._random_states is not _old_cupy_random_states
    if not deterministic:
        random.seed()
        numpy.random.seed()
        cupy.random.seed()
    else:
        random.seed(99)
        numpy.random.seed(100)
        cupy.random.seed(101)