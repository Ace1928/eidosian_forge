import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def post_dominators(self):
    """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes post-dominating it.

        A node P post-dominates a node N when any path starting from N must go
        through P.
        """
    return self._post_doms