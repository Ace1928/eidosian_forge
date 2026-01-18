import abc
import weakref
from numba.core import errors
def save_cache(self, orig_disp, new_disp):
    """Save a dispatcher associated with the given key.
        """
    self._cache[orig_disp] = new_disp