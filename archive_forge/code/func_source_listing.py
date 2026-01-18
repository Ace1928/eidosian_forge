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
def source_listing(self, parfors_simple, purpose_str):
    filename = self.func_ir.loc.filename
    count = self.count_parfors()
    func_name = self.func_ir.func_id.func
    try:
        lines = inspect.getsource(func_name).splitlines()
    except OSError:
        lines = None
    if lines and parfors_simple:
        src_width = max([len(x) for x in lines])
        map_line_to_pf = defaultdict(list)
        for k, v in parfors_simple.items():
            if parfors_simple[k][1].filename == filename:
                match_line = self.sort_pf_by_line(k, parfors_simple)
                map_line_to_pf[match_line].append(str(k))
        max_pf_per_line = max([1] + [len(x) for x in map_line_to_pf.values()])
        width = src_width + (1 + max_pf_per_line * (len(str(count)) + 2))
        newlines = []
        newlines.append('\n')
        newlines.append('Parallel loop listing for %s' % purpose_str)
        newlines.append(width * '-' + '|loop #ID')
        fmt = '{0:{1}}| {2}'
        lstart = max(0, self.func_ir.loc.line - 1)
        for no, line in enumerate(lines, lstart):
            pf_ids = map_line_to_pf.get(no, None)
            if pf_ids is not None:
                pfstr = '#' + ', '.join(pf_ids)
            else:
                pfstr = ''
            stripped = line.strip('\n')
            srclen = len(stripped)
            if pf_ids:
                l = fmt.format(width * '-', width, pfstr)
            else:
                l = fmt.format(width * ' ', width, pfstr)
            newlines.append(stripped + l[srclen:])
        print('\n'.join(newlines))
    else:
        print('No source available')