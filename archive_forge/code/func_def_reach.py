import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def def_reach(dct):
    """Find all variable definition reachable at the entry of a block
        """
    for offset in var_def_map:
        used_or_defined = var_def_map[offset] | var_use_map[offset]
        dct[offset] |= used_or_defined
        for out_blk, _ in cfg.successors(offset):
            dct[out_blk] |= dct[offset]