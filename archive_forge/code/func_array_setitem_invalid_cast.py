import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def array_setitem_invalid_cast():
    arr = np.empty(1, dtype=np.float64)
    arr[0] = 1j
    return arr