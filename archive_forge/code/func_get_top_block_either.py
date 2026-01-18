import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def get_top_block_either(self, *kinds):
    """Find the first block that matches *kind*
        """
    kinds = {BlockKind(kind) for kind in kinds}
    for bs in reversed(self._blockstack):
        if bs['kind'] in kinds:
            return bs