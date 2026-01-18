import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
@property
def outgoing_phis(self):
    """The dictionary of outgoing phi nodes.

        The keys are the name of the PHI nodes.
        The values are the outgoing states.
        """
    return self._outgoing_phis