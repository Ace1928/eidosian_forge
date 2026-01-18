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
def list_unify_usecase2(n):
    res = []
    for i in range(n):
        if i & 1:
            res.append((i, 1.0))
        else:
            res.append((2.0, i))
    res.append((123j, 42))
    return res