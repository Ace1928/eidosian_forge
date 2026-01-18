import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def exit_points(self):
    """
        Return the computed set of exit nodes (may be empty).
        """
    return self._exit_points