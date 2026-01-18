import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def find_use_defs():
    defmap = {}
    phismap = defaultdict(set)
    for state in runner.finished:
        for phi, rhs in state._outgoing_phis.items():
            if rhs not in phi_set:
                defmap[phi] = state
            phismap[phi].add((rhs, state))
    _logger.debug('defmap: %s', _lazy_pformat(defmap))
    _logger.debug('phismap: %s', _lazy_pformat(phismap))
    return (defmap, phismap)