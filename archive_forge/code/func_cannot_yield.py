from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def cannot_yield(loop):
    """cannot have yield inside the loop"""
    insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
    for blk in map(blocks.__getitem__, insiders):
        for inst in blk.body:
            if isinstance(inst, ir.Assign):
                if isinstance(inst.value, ir.Yield):
                    _logger.debug('has yield')
                    return False
    _logger.debug('no yield')
    return True