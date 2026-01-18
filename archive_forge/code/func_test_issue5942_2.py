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
def test_issue5942_2(self):

    def test_impl(d, k):
        gg = np.zeros((d, k), dtype=np.int32)
        gg_next = np.zeros((d, k), dtype=np.int32)
        for i_gg in prange(d):
            for n in range(k):
                gg[i_gg, n] = i_gg
            gg_next[i_gg, :] = gg[i_gg, :]
            gg_next[i_gg, 0] += 1
        return gg_next
    d = 4
    k = 2
    self.check(test_impl, d, k)