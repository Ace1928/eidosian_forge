import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def copy_propagate(blocks, typemap):
    """compute copy propagation information for each block using fixed-point
     iteration on data flow equations:
     in_b = intersect(predec(B))
     out_b = gen_b | (in_b - kill_b)
    """
    cfg = compute_cfg_from_blocks(blocks)
    entry = cfg.entry_point()
    c_data = init_copy_propagate_data(blocks, entry, typemap)
    gen_copies, all_copies, kill_copies, in_copies, out_copies = c_data
    old_point = None
    new_point = copy.deepcopy(out_copies)
    while old_point != new_point:
        for label in blocks.keys():
            if label == entry:
                continue
            predecs = [i for i, _d in cfg.predecessors(label)]
            in_copies[label] = out_copies[predecs[0]].copy()
            for p in predecs:
                in_copies[label] &= out_copies[p]
            out_copies[label] = gen_copies[label] | in_copies[label] - kill_copies[label]
        old_point = new_point
        new_point = copy.deepcopy(out_copies)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('copy propagate out_copies:', out_copies)
    return (in_copies, out_copies)