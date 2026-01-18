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
def init_copy_propagate_data(blocks, entry, typemap):
    """get initial condition of copy propagation data flow for each block.
    """
    gen_copies, extra_kill = get_block_copies(blocks, typemap)
    all_copies = set()
    for l, s in gen_copies.items():
        all_copies |= gen_copies[l]
    kill_copies = {}
    for label, gen_set in gen_copies.items():
        kill_copies[label] = set()
        for lhs, rhs in all_copies:
            if lhs in extra_kill[label] or rhs in extra_kill[label]:
                kill_copies[label].add((lhs, rhs))
            assigned = {lhs for lhs, rhs in gen_set}
            if (lhs, rhs) not in gen_set and (lhs in assigned or rhs in assigned):
                kill_copies[label].add((lhs, rhs))
    in_copies = {l: all_copies.copy() for l in blocks.keys()}
    in_copies[entry] = set()
    out_copies = {}
    for label in blocks.keys():
        out_copies[label] = gen_copies[label] | in_copies[label] - kill_copies[label]
    out_copies[entry] = gen_copies[entry]
    return (gen_copies, all_copies, kill_copies, in_copies, out_copies)