import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def fix_point_progress():
    return tuple(map(len, block_entry_vars.values()))