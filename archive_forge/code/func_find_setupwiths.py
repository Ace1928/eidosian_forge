from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def find_setupwiths(func_ir):
    """Find all top-level with.

    Returns a list of ranges for the with-regions.
    """

    def find_ranges(blocks):
        cfg = compute_cfg_from_blocks(blocks)
        sus_setups, sus_pops = (set(), set())
        for label, block in blocks.items():
            for stmt in block.body:
                if ir_utils.is_setup_with(stmt):
                    sus_setups.add(label)
                if ir_utils.is_pop_block(stmt):
                    sus_pops.add(label)
        setup_with_to_pop_blocks_map = defaultdict(set)
        for setup_block in cfg.topo_sort(sus_setups, reverse=True):
            to_visit, seen = ([], [])
            to_visit.append(setup_block)
            while to_visit:
                block = to_visit.pop()
                seen.append(block)
                for stmt in blocks[block].body:
                    if ir_utils.is_raise(stmt):
                        raise errors.CompilerError('unsupported control flow due to raise statements inside with block')
                    if ir_utils.is_pop_block(stmt) and block in sus_pops:
                        setup_with_to_pop_blocks_map[setup_block].add(block)
                        sus_pops.remove(block)
                        break
                    if ir_utils.is_terminator(stmt):
                        for t in stmt.get_targets():
                            if t not in seen:
                                to_visit.append(t)
        return setup_with_to_pop_blocks_map
    blocks = func_ir.blocks
    with_ranges_dict = find_ranges(blocks)
    func_ir = consolidate_multi_exit_withs(with_ranges_dict, blocks, func_ir)
    with_ranges_tuple = [(s, list(p)[0]) for s, p in with_ranges_dict.items()]
    for _, p in with_ranges_tuple:
        targets = blocks[p].terminator.get_targets()
        if len(targets) != 1:
            raise errors.CompilerError('unsupported control flow: with-context contains branches (i.e. break/return/raise) that can leave the block ')
    for _, p in with_ranges_tuple:
        target_block = blocks[p]
        if ir_utils.is_return(func_ir.blocks[target_block.terminator.get_targets()[0]].terminator):
            _rewrite_return(func_ir, p)
    with_ranges_tuple = [(s, func_ir.blocks[p].terminator.get_targets()[0]) for s, p in with_ranges_tuple]
    with_ranges_tuple = _eliminate_nested_withs(with_ranges_tuple)
    return (with_ranges_tuple, func_ir)