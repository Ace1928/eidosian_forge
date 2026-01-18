import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def prune_by_value(branch, condition, blk, *conds):
    lhs_cond, rhs_cond = conds
    try:
        take_truebr = condition.fn(lhs_cond, rhs_cond)
    except Exception:
        return (False, None)
    if DEBUG > 0:
        kill = branch.falsebr if take_truebr else branch.truebr
        print('Pruning %s' % kill, branch, lhs_cond, rhs_cond, condition.fn)
    taken = do_prune(take_truebr, blk)
    return (True, taken)