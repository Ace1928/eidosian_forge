import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_DELETE_SLICE_3(self, state, inst):
    """
        del TOS2[TOS1:TOS]
        """
    tos = state.pop()
    tos1 = state.pop()
    tos2 = state.pop()
    slicevar = state.make_temp()
    indexvar = state.make_temp()
    state.append(inst, base=tos2, start=tos1, stop=tos, slicevar=slicevar, indexvar=indexvar)