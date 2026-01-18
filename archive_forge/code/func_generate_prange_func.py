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
def generate_prange_func(self, pyfunc, patch_instance):
    """
        This function does the actual code augmentation to enable the explicit
        testing of `prange` calls in place of `range`.
        """
    pyfunc_code = pyfunc.__code__
    prange_names = list(pyfunc_code.co_names)
    if patch_instance is None:
        assert 'range' in pyfunc_code.co_names
        prange_names = tuple([x if x != 'range' else 'prange' for x in pyfunc_code.co_names])
        new_code = bytes(pyfunc_code.co_code)
    else:
        range_idx = pyfunc_code.co_names.index('range')
        range_locations = []
        for instr in dis.Bytecode(pyfunc_code):
            if instr.opname == 'LOAD_GLOBAL':
                if _fix_LOAD_GLOBAL_arg(instr.arg) == range_idx:
                    range_locations.append(instr.offset + 1)
        prange_names.append('prange')
        prange_names = tuple(prange_names)
        prange_idx = len(prange_names) - 1
        if utils.PYVERSION in ((3, 11), (3, 12)):
            prange_idx = 1 + (prange_idx << 1)
        elif utils.PYVERSION in ((3, 9), (3, 10)):
            pass
        else:
            raise NotImplementedError(utils.PYVERSION)
        new_code = bytearray(pyfunc_code.co_code)
        assert len(patch_instance) <= len(range_locations)
        for i in patch_instance:
            idx = range_locations[i]
            new_code[idx] = prange_idx
        new_code = bytes(new_code)
    prange_code = pyfunc_code.replace(co_code=new_code, co_names=prange_names)
    pfunc = pytypes.FunctionType(prange_code, globals())
    return pfunc