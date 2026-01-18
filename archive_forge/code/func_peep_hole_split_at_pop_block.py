import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def peep_hole_split_at_pop_block(func_ir):
    """
    Split blocks that contain ir.PopBlock.

    This rewrite restores the IR structure to pre 3.11 so that withlifting
    can work correctly.
    """
    new_block_map = {}
    sorted_blocks = sorted(func_ir.blocks.items())
    for blk_idx, (label, blk) in enumerate(sorted_blocks):
        pop_block_locs = []
        for i, inst in enumerate(blk.body):
            if isinstance(inst, ir.PopBlock):
                pop_block_locs.append(i)
        if pop_block_locs:
            new_blocks = []
            for i in pop_block_locs:
                before_blk = ir.Block(blk.scope, loc=blk.loc)
                before_blk.body.extend(blk.body[:i])
                new_blocks.append(before_blk)
                popblk_blk = ir.Block(blk.scope, loc=blk.loc)
                popblk_blk.body.append(blk.body[i])
                new_blocks.append(popblk_blk)
            prev_label = label
            for newblk in new_blocks:
                new_block_map[prev_label] = newblk
                next_label = prev_label + 1
                newblk.body.append(ir.Jump(next_label, loc=blk.loc))
                prev_label = next_label
            if blk_idx + 1 < len(sorted_blocks):
                if prev_label >= sorted_blocks[blk_idx + 1][0]:
                    raise errors.InternalError('POP_BLOCK peephole failed')
            tail_blk = ir.Block(blk.scope, loc=blk.loc)
            tail_blk.body.extend(blk.body[pop_block_locs[-1] + 1:])
            new_block_map[prev_label] = tail_blk
    func_ir.blocks.update(new_block_map)
    return func_ir