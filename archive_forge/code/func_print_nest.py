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
def print_nest(fadj_, nadj_, theroot, reported, region_id):

    def print_g(fadj_, nadj_, nroot, depth):
        for k in nadj_[nroot]:
            msg = fac * depth * ' ' + '%s%s %s' % (sword, k, '(serial')
            if nadj_[k] == []:
                fused = []
                if fadj_[k] != [] and k not in reported:
                    fused = sorted(self.reachable_nodes(fadj_, k))
                    msg += ', fused with loop(s): '
                    msg += ', '.join([str(x) for x in fused])
                msg += ')'
                reported.append(k)
                print_wrapped(msg)
                summary[region_id]['fused'] += len(fused)
            else:
                print_wrapped(msg + ')')
                print_g(fadj_, nadj_, k, depth + 1)
            summary[region_id]['serialized'] += 1
    if nadj_[theroot] != []:
        print_wrapped('Parallel region %s:' % region_id)
        print_wrapped('%s%s %s' % (sword, theroot, '(parallel)'))
        summary[region_id] = {'root': theroot, 'fused': 0, 'serialized': 0}
        print_g(fadj_, nadj_, theroot, 1)
        print('\n')
        region_id = region_id + 1
    return region_id