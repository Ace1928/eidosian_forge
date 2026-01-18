from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def has_multiple_backedges(loop):
    count = 0
    for k in loop.body:
        blk = blocks[k]
        edges = blk.terminator.get_targets()
        if loop.header in edges:
            count += 1
            if count > 1:
                return True
    return False