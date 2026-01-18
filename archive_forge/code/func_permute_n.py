import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def permute_n(n):
    if n == 1:
        return ([0],)
    ret = ()
    base = permute_n(n - 1)
    for perm in base:
        for i in range(n):
            new = perm + [n - 1]
            new[n - 1] = new[i]
            new[i] = n - 1
            ret += (new,)
    return ret