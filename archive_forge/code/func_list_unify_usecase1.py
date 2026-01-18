import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def list_unify_usecase1(n):
    res = 0
    x = []
    if n < 10:
        x.append(np.int32(n))
    else:
        for i in range(n):
            x.append(np.int64(i))
    x.append(5.0)
    for j in range(len(x)):
        res += j * x[j]
    for val in x:
        res += int(val) & len(x)
    while len(x) > 0:
        res += x.pop()
    return res