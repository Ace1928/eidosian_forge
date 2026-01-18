import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def compute_live_map(cfg, blocks, var_use_map, var_def_map):
    """
    Find variables that must be alive at the ENTRY of each block.
    We use a simple fix-point algorithm that iterates until the set of
    live variables is unchanged for each block.
    """

    def fix_point_progress(dct):
        """Helper function to determine if a fix-point has been reached.
        """
        return tuple((len(v) for v in dct.values()))

    def fix_point(fn, dct):
        """Helper function to run fix-point algorithm.
        """
        old_point = None
        new_point = fix_point_progress(dct)
        while old_point != new_point:
            fn(dct)
            old_point = new_point
            new_point = fix_point_progress(dct)

    def def_reach(dct):
        """Find all variable definition reachable at the entry of a block
        """
        for offset in var_def_map:
            used_or_defined = var_def_map[offset] | var_use_map[offset]
            dct[offset] |= used_or_defined
            for out_blk, _ in cfg.successors(offset):
                dct[out_blk] |= dct[offset]

    def liveness(dct):
        """Find live variables.

        Push var usage backward.
        """
        for offset in dct:
            live_vars = dct[offset]
            for inc_blk, _data in cfg.predecessors(offset):
                reachable = live_vars & def_reach_map[inc_blk]
                dct[inc_blk] |= reachable - var_def_map[inc_blk]
    live_map = {}
    for offset in blocks.keys():
        live_map[offset] = set(var_use_map[offset])
    def_reach_map = defaultdict(set)
    fix_point(def_reach, def_reach_map)
    fix_point(liveness, live_map)
    return live_map