import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def arange_parallel_impl(return_type, *args, dtype=None):
    inferred_dtype = as_dtype(return_type.dtype)

    def arange_1(stop):
        return np.arange(0, stop, 1, inferred_dtype)

    def arange_1_dtype(stop, dtype):
        return np.arange(0, stop, 1, dtype)

    def arange_2(start, stop):
        return np.arange(start, stop, 1, inferred_dtype)

    def arange_2_dtype(start, stop, dtype):
        return np.arange(start, stop, 1, dtype)

    def arange_3(start, stop, step):
        return np.arange(start, stop, step, inferred_dtype)

    def arange_3_dtype(start, stop, step, dtype):
        return np.arange(start, stop, step, dtype)
    if any((isinstance(a, types.Complex) for a in args)):

        def arange_4(start, stop, step, dtype):
            numba.parfors.parfor.init_prange()
            nitems_c = (stop - start) / step
            nitems_r = math.ceil(nitems_c.real)
            nitems_i = math.ceil(nitems_c.imag)
            nitems = int(max(min(nitems_i, nitems_r), 0))
            arr = np.empty(nitems, dtype)
            for i in numba.parfors.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr
    else:

        def arange_4(start, stop, step, dtype):
            numba.parfors.parfor.init_prange()
            nitems_r = math.ceil((stop - start) / step)
            nitems = int(max(nitems_r, 0))
            arr = np.empty(nitems, dtype)
            val = start
            for i in numba.parfors.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr
    if len(args) == 1:
        return arange_1 if dtype is None else arange_1_dtype
    elif len(args) == 2:
        return arange_2 if dtype is None else arange_2_dtype
    elif len(args) == 3:
        return arange_3 if dtype is None else arange_3_dtype
    elif len(args) == 4:
        return arange_4
    else:
        raise ValueError('parallel arange with types {}'.format(args))