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
def get_copies_parfor(parfor, typemap):
    """find copies generated/killed by parfor"""
    blocks = wrap_parfor_blocks(parfor)
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks, typemap)
    in_gen_copies, in_extra_kill = get_block_copies(blocks, typemap)
    unwrap_parfor_blocks(parfor)
    kill_set = in_extra_kill[0]
    for label in parfor.loop_body.keys():
        kill_set |= {l for l, r in in_gen_copies[label]}
        kill_set |= in_extra_kill[label]
    last_label = max(parfor.loop_body.keys())
    gens = out_copies_parfor[last_label] & in_gen_copies[0]
    if config.DEBUG_ARRAY_OPT >= 1:
        print('copy propagate parfor gens:', gens, 'kill_set', kill_set)
    return (gens, kill_set)