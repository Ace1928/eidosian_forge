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
def copy_args(*args):
    if not args:
        return tuple()
    new_args = []
    for x in args:
        if isinstance(x, np.ndarray):
            new_args.append(x.copy('k'))
        elif isinstance(x, np.number):
            new_args.append(x.copy())
        elif isinstance(x, numbers.Number):
            new_args.append(x)
        elif x is None:
            new_args.append(x)
        elif isinstance(x, tuple):
            new_args.append(copy.deepcopy(x))
        elif isinstance(x, list):
            new_args.append(x[:])
        else:
            raise ValueError('Unsupported argument type encountered')
    return tuple(new_args)