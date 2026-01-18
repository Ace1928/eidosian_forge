import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_ROT_THREE(self, state, inst):
    first = state.pop()
    second = state.pop()
    third = state.pop()
    state.push(first)
    state.push(third)
    state.push(second)