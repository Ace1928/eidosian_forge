import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def test_dead_randoms(self):

    def test_impl(n):
        A = np.random.standard_normal(size=(n, n))
        B = np.random.randn(n, n)
        C = np.random.normal(0.0, 1.0, (n, n))
        D = np.random.chisquare(1.0, (n, n))
        E = np.random.randint(1, high=3, size=(n, n))
        F = np.random.triangular(1, 2, 3, (n, n))
        return 3
    n = 128
    cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
    parfor_output = cpfunc.entry_point(n)
    py_output = test_impl(n)
    self.assertEqual(parfor_output, py_output)
    self.assertEqual(countParfors(test_impl, (types.int64,)), 0)