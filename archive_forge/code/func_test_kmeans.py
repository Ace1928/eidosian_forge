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
def test_kmeans(self):
    np.random.seed(0)
    N = 1024
    D = 10
    centers = 3
    A = np.random.ranf((N, D))
    init_centroids = np.random.ranf((centers, D))
    self.check(example_kmeans_test, A, centers, 3, init_centroids, decimal=1)
    arg_typs = (types.Array(types.float64, 2, 'C'), types.intp, types.intp, types.Array(types.float64, 2, 'C'))
    self.assertEqual(countNonParforArrayAccesses(example_kmeans_test, arg_typs), 0)