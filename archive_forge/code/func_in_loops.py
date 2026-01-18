import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def in_loops(self, node):
    """
        Return the list of Loop objects the *node* belongs to,
        from innermost to outermost.
        """
    return [self._loops[x] for x in self._in_loops.get(node, ())]