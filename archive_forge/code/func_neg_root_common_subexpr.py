import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def neg_root_common_subexpr(As, Bs, Cs):
    _2As = 2.0 * As
    _4AsCs = 2.0 * _2As * Cs
    _Bs2_4AsCs = Bs ** 2.0 - _4AsCs
    return (-Bs - _Bs2_4AsCs ** 0.5) / _2As