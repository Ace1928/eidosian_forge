import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BEGIN_FINALLY(self, state, inst):
    temps = []
    for i in range(_EXCEPT_STACK_OFFSET):
        tmp = state.make_temp()
        temps.append(tmp)
        state.push(tmp)
    state.append(inst, temps=temps)