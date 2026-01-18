import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def get_timers(fn, prop):
    md = fn.get_metadata(fn.signatures[0])
    return md[prop]