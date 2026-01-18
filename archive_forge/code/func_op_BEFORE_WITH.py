import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BEFORE_WITH(self, state, inst):
    cm = state.pop()
    yielded = state.make_temp()
    exitfn = state.make_temp(prefix='setup_with_exitfn')
    state.push(exitfn)
    state.push(yielded)
    bc = state._bytecode
    ehhead = bc.find_exception_entry(inst.next)
    ehrelated = [ehhead]
    for eh in bc.exception_entries:
        if eh.target == ehhead.target:
            ehrelated.append(eh)
    end = max((eh.end for eh in ehrelated))
    state.append(inst, contextmanager=cm, exitfn=exitfn, end=end)
    state.push_block(state.make_block(kind='WITH', end=end))
    state.fork(pc=inst.next)