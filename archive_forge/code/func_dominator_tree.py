import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def dominator_tree(self):
    """
        return a dictionary of {node -> set(nodes)} mapping each node to
        the set of nodes it immediately dominates

        The domtree(B) is the closest strict set of nodes that B dominates
        """
    return self._domtree