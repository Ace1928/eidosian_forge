import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def do_teardown():
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    random.setstate(_old_python_random_state)
    numpy.random.set_state(_old_numpy_random_state)
    cupy.random._generator._random_states = _old_cupy_random_states
    _old_python_random_state = None
    _old_numpy_random_state = None
    _old_cupy_random_states = None