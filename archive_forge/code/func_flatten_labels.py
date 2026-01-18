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
def flatten_labels(blocks):
    """makes the labels in range(0, len(blocks)), useful to compare CFGs
    """
    blocks = add_offset_to_labels(blocks, find_max_label(blocks) + 1)
    new_blocks = {}
    topo_order = find_topo_order(blocks)
    l_map = dict()
    idx = 0
    for x in topo_order:
        l_map[x] = idx
        idx += 1
    for t_node in topo_order:
        b = blocks[t_node]
        term = None
        if b.body:
            term = b.body[-1]
        if isinstance(term, ir.Jump):
            b.body[-1] = ir.Jump(l_map[term.target], term.loc)
        if isinstance(term, ir.Branch):
            b.body[-1] = ir.Branch(term.cond, l_map[term.truebr], l_map[term.falsebr], term.loc)
        new_blocks[l_map[t_node]] = b
    return new_blocks