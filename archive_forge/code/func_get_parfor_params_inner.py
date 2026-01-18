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
def get_parfor_params_inner(parfor, pre_defs, options_fusion, fusion_info):
    blocks = wrap_parfor_blocks(parfor)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    parfor_ids, _ = get_parfor_params(blocks, options_fusion, fusion_info)
    n_parfors = len(parfor_ids)
    if n_parfors > 0:
        if config.DEBUG_ARRAY_OPT_STATS:
            after_fusion = 'After fusion' if options_fusion else 'With fusion disabled'
            print('{}, parallel for-loop {} has nested Parfor(s) #{}.'.format(after_fusion, parfor.id, n_parfors, parfor_ids))
        fusion_info[parfor.id] = list(parfor_ids)
    unwrap_parfor_blocks(parfor)
    keylist = sorted(live_map.keys())
    init_block = keylist[0]
    first_non_init_block = keylist[1]
    before_defs = usedefs.defmap[init_block] | pre_defs
    params = live_map[first_non_init_block] & before_defs
    return params