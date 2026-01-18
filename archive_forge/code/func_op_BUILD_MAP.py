import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BUILD_MAP(self, state, inst):
    dct = state.make_temp()
    count = inst.arg
    items = []
    for i in range(count):
        v, k = (state.pop(), state.pop())
        items.append((k, v))
    state.append(inst, items=items[::-1], size=count, res=dct)
    state.push(dct)