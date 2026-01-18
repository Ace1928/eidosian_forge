from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def canonicalize_cfg_single_backedge(blocks):
    """
    Rewrite loops that have multiple backedges.
    """
    cfg = compute_cfg_from_blocks(blocks)
    newblocks = blocks.copy()

    def new_block_id():
        return max(newblocks.keys()) + 1

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

    def yield_loops_with_multiple_backedges():
        for lp in cfg.loops().values():
            if has_multiple_backedges(lp):
                yield lp

    def replace_target(term, src, dst):

        def replace(target):
            return dst if target == src else target
        if isinstance(term, ir.Branch):
            return ir.Branch(cond=term.cond, truebr=replace(term.truebr), falsebr=replace(term.falsebr), loc=term.loc)
        elif isinstance(term, ir.Jump):
            return ir.Jump(target=replace(term.target), loc=term.loc)
        else:
            assert not term.get_targets()
            return term

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
    for loop in yield_loops_with_multiple_backedges():
        rewrite_single_backedge(loop)
    return newblocks