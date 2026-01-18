import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_BUILD_STRING(self, state, inst):
    """
        BUILD_STRING(count): Concatenates count strings from the stack and
        pushes the resulting string onto the stack.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
        """
    count = inst.arg
    strings = list(reversed([state.pop() for _ in range(count)]))
    if count == 0:
        tmps = [state.make_temp()]
    else:
        tmps = [state.make_temp() for _ in range(count - 1)]
    state.append(inst, strings=strings, tmps=tmps)
    state.push(tmps[-1])