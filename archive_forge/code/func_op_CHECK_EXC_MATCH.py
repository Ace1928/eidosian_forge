import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_CHECK_EXC_MATCH(self, state, inst):
    pred = state.make_temp('predicate')
    tos = state.pop()
    tos1 = state.get_tos()
    state.append(inst, pred=pred, tos=tos, tos1=tos1)
    state.push(pred)