from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def get_arraytypes():
    """pygame.surfarray.get_arraytypes(): return tuple

    DEPRECATED - only numpy arrays are now supported.
    """
    warnings.warn(DeprecationWarning('only numpy arrays are now supported, this function will be removed in a future version of the module'))
    return ('numpy',)