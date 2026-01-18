import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def descendents(self, node):
    """
        Return the set of descendents of the given *node*, in topological
        order (ignoring back edges).
        """
    return self._descs[node]