import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def get_outgoing_states(self):
    """Get states for each outgoing edges
        """
    assert not self._outgoing_phis
    ret = []
    for edge in self._outedges:
        state = State(bytecode=self._bytecode, pc=edge.pc, nstack=len(edge.stack), blockstack=edge.blockstack, nullvals=[i for i, v in enumerate(edge.stack) if _is_null_temp_reg(v)])
        ret.append(state)
        for phi, i in state._phis.items():
            self._outgoing_phis[phi] = edge.stack[i]
    return ret