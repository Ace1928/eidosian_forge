from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def slice_constructor(*args):
    sl = slice(*args)
    return (sl.start, sl.stop, sl.step)