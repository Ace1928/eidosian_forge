import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def dead_nodes(self):
    """
        Return the set of dead nodes (eliminated from the graph).
        """
    return self._dead_nodes