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
def simplify_CFG(blocks):
    """transform chains of blocks that have no loop into a single block"""
    cfg = compute_cfg_from_blocks(blocks)

    def find_single_branch(label):
        block = blocks[label]
        return len(block.body) == 1 and isinstance(block.body[0], ir.Branch)
    single_branch_blocks = list(filter(find_single_branch, blocks.keys()))
    marked_for_del = set()
    for label in single_branch_blocks:
        inst = blocks[label].body[0]
        predecessors = cfg.predecessors(label)
        delete_block = True
        for p, q in predecessors:
            block = blocks[p]
            if isinstance(block.body[-1], ir.Jump):
                block.body[-1] = copy.copy(inst)
            else:
                delete_block = False
        if delete_block:
            marked_for_del.add(label)
    for label in marked_for_del:
        del blocks[label]
    merge_adjacent_blocks(blocks)
    return rename_labels(blocks)