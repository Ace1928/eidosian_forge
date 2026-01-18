import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def return_rjust2(x, i, w, y, j):
    return x[i].rjust(w, y[j])