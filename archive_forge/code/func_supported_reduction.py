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
def supported_reduction(x, func_ir):
    if x.op == 'inplace_binop' or x.op == 'binop':
        if x.fn == operator.ifloordiv or x.fn == operator.floordiv:
            raise errors.NumbaValueError('Parallel floordiv reductions are not supported. If all divisors are integers then a floordiv reduction can in some cases be parallelized as a multiply reduction followed by a floordiv of the resulting product.', x.loc)
        supps = [operator.iadd, operator.isub, operator.imul, operator.itruediv, operator.add, operator.sub, operator.mul, operator.truediv]
        return x.fn in supps
    if x.op == 'call':
        callname = guard(find_callname, func_ir, x)
        if callname in [('max', 'builtins'), ('min', 'builtins'), ('datetime_minimum', 'numba.np.npdatetime_helpers'), ('datetime_maximum', 'numba.np.npdatetime_helpers')]:
            return True
    return False