import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def get_used_phis_per_state():
    used_phis = defaultdict(set)
    phi_set = set()
    for state in runner.finished:
        used = set(state._used_regs)
        phis = set(state._phis)
        used_phis[state] |= phis & used
        phi_set |= phis
    return (used_phis, phi_set)