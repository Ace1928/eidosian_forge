import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_POP_FINALLY(self, state, inst):
    if inst.arg != 0:
        msg = 'Unsupported use of a bytecode related to try..finally or a with-context'
        raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))