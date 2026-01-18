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
def fuse_parfors_inner(parfor1, parfor2):
    parfor1.init_block.body.extend(parfor2.init_block.body)
    parfor2_first_label = min(parfor2.loop_body.keys())
    parfor2_first_block = parfor2.loop_body[parfor2_first_label].body
    parfor1_first_label = min(parfor1.loop_body.keys())
    parfor1_last_label = max(parfor1.loop_body.keys())
    parfor1.loop_body[parfor1_last_label].body.extend(parfor2_first_block)
    parfor1.loop_body.update(parfor2.loop_body)
    parfor1.loop_body.pop(parfor2_first_label)
    ndims = len(parfor1.loop_nests)
    index_dict = {parfor2.index_var.name: parfor1.index_var}
    for i in range(ndims):
        index_dict[parfor2.loop_nests[i].index_variable.name] = parfor1.loop_nests[i].index_variable
    replace_vars(parfor1.loop_body, index_dict)
    blocks = wrap_parfor_blocks(parfor1, entry_label=parfor1_first_label)
    blocks = rename_labels(blocks)
    unwrap_parfor_blocks(parfor1, blocks)
    nameset = set((x.name for x in index_dict.values()))
    remove_duplicate_definitions(parfor1.loop_body, nameset)
    parfor1.patterns.extend(parfor2.patterns)
    if config.DEBUG_ARRAY_OPT_STATS:
        print('Parallel for-loop #{} is fused into for-loop #{}.'.format(parfor2.id, parfor1.id))
    msg = '- fusion succeeded: parallel for-loop #{} is fused into for-loop #{}.'
    msg = msg.format(parfor2.id, parfor1.id)
    report = FusionReport(parfor1.id, parfor2.id, msg)
    return (parfor1, report)