import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def liveness(dct):
    """Find live variables.

        Push var usage backward.
        """
    for offset in dct:
        live_vars = dct[offset]
        for inc_blk, _data in cfg.predecessors(offset):
            reachable = live_vars & def_reach_map[inc_blk]
            dct[inc_blk] |= reachable - var_def_map[inc_blk]