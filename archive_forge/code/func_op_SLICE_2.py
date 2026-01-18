import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_SLICE_2(self, state, inst):
    """
        TOS = TOS1[:TOS]
        """
    tos = state.pop()
    tos1 = state.pop()
    res = state.make_temp()
    slicevar = state.make_temp()
    indexvar = state.make_temp()
    nonevar = state.make_temp()
    state.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)
    state.push(res)