import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_SETUP_WITH(self, state, inst):
    cm = state.pop()
    yielded = state.make_temp()
    exitfn = state.make_temp(prefix='setup_with_exitfn')
    state.append(inst, contextmanager=cm, exitfn=exitfn)
    if PYVERSION < (3, 9):
        state.push_block(state.make_block(kind='WITH_FINALLY', end=inst.get_jump_target()))
    state.push(exitfn)
    state.push(yielded)
    state.push_block(state.make_block(kind='WITH', end=inst.get_jump_target()))
    state.fork(pc=inst.next)