from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def sorted_utility_codes_and_deps(utilcodes):
    ranks = {}
    get_rank = ranks.get

    def calculate_rank(utilcode):
        rank = get_rank(utilcode)
        if rank is None:
            ranks[utilcode] = 0
            original_order = len(ranks)
            rank = ranks[utilcode] = 1 + (min([calculate_rank(dep) for dep in utilcode.requires]) if utilcode.requires else -1) + original_order * 1e-08
        return rank
    for utilcode in utilcodes:
        calculate_rank(utilcode)
    return sorted(ranks, key=get_rank)