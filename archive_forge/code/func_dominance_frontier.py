import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def dominance_frontier(self):
    """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes in its dominance frontier.

        The dominance frontier _df(N) is the set of all nodes that are
        immediate successors to blocks dominated by N but which aren't
        strictly dominated by N
        """
    return self._df