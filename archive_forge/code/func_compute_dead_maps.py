import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def compute_dead_maps(cfg, blocks, live_map, var_def_map):
    """
    Compute the end-of-live information for variables.
    `live_map` contains a mapping of block offset to all the living
    variables at the ENTRY of the block.
    """
    escaping_dead_map = defaultdict(set)
    internal_dead_map = defaultdict(set)
    exit_dead_map = defaultdict(set)
    for offset, ir_block in blocks.items():
        cur_live_set = live_map[offset] | var_def_map[offset]
        outgoing_live_map = dict(((out_blk, live_map[out_blk]) for out_blk, _data in cfg.successors(offset)))
        terminator_liveset = set((v.name for v in ir_block.terminator.list_vars()))
        combined_liveset = reduce(operator.or_, outgoing_live_map.values(), set())
        combined_liveset |= terminator_liveset
        internal_set = cur_live_set - combined_liveset
        internal_dead_map[offset] = internal_set
        escaping_live_set = cur_live_set - internal_set
        for out_blk, new_live_set in outgoing_live_map.items():
            new_live_set = new_live_set | var_def_map[out_blk]
            escaping_dead_map[out_blk] |= escaping_live_set - new_live_set
        if not outgoing_live_map:
            exit_dead_map[offset] = terminator_liveset
    all_vars = reduce(operator.or_, live_map.values(), set())
    internal_dead_vars = reduce(operator.or_, internal_dead_map.values(), set())
    escaping_dead_vars = reduce(operator.or_, escaping_dead_map.values(), set())
    exit_dead_vars = reduce(operator.or_, exit_dead_map.values(), set())
    dead_vars = internal_dead_vars | escaping_dead_vars | exit_dead_vars
    missing_vars = all_vars - dead_vars
    if missing_vars:
        if not cfg.exit_points():
            pass
        else:
            msg = 'liveness info missing for vars: {0}'.format(missing_vars)
            raise RuntimeError(msg)
    combined = dict(((k, internal_dead_map[k] | escaping_dead_map[k]) for k in blocks))
    return _dead_maps_result(internal=internal_dead_map, escaping=escaping_dead_map, combined=combined)