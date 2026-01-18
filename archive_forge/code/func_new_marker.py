import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle
def new_marker(self, name):
    """Create new Marker object owned by this domain

        Parameters
        ----------
        name : string
            Name of the marker
        """
    return Marker(self, name)