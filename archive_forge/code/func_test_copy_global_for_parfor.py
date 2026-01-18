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
def test_copy_global_for_parfor(self):
    """ issue4903: a global is copied next to a parfor so that
            it can be inlined into the parfor and thus not have to be
            passed to the parfor (i.e., an unsupported function type).
            This global needs to be renamed in the block into which
            it is copied.
        """

    def test_impl(zz, tc):
        lh = np.zeros(len(tc))
        lc = np.zeros(len(tc))
        for i in range(1):
            nt = tc[i]
            for t in range(nt):
                lh += np.exp(zz[i, t])
            for t in range(nt):
                lc += np.exp(zz[i, t])
        return (lh, lc)
    m = 2
    zz = np.ones((m, m, m))
    tc = np.ones(m, dtype=np.int_)
    self.prange_tester(test_impl, zz, tc, patch_instance=[0])