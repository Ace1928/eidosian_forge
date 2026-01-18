import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_POP_EXCEPT(self, state, inst):
    blk = state.pop_block()
    if blk['kind'] not in {BlockKind('EXCEPT'), BlockKind('FINALLY')}:
        raise UnsupportedError(f'POP_EXCEPT got an unexpected block: {blk['kind']}', loc=self.get_debug_loc(inst.lineno))
    state.pop()
    state.pop()
    state.pop()
    state.fork(pc=inst.next)