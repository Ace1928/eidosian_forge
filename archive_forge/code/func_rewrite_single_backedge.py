from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def rewrite_single_backedge(loop):
    """
        Add new tail block that gathers all the backedges
        """
    header = loop.header
    tailkey = new_block_id()
    for blkkey in loop.body:
        blk = newblocks[blkkey]
        if header in blk.terminator.get_targets():
            newblk = blk.copy()
            newblk.body[-1] = replace_target(blk.terminator, header, tailkey)
            newblocks[blkkey] = newblk
    entryblk = newblocks[header]
    tailblk = ir.Block(scope=entryblk.scope, loc=entryblk.loc)
    tailblk.append(ir.Jump(target=header, loc=tailblk.loc))
    newblocks[tailkey] = tailblk