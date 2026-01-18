from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def blit_array(surface, array):
    """pygame.surfarray.blit_array(Surface, array): return None

    Blit directly from a array values.

    Directly copy values from an array into a Surface. This is faster than
    converting the array into a Surface and blitting. The array must be the
    same dimensions as the Surface and will completely replace all pixel
    values. Only integer, ascii character and record arrays are accepted.

    This function will temporarily lock the Surface as the new values are
    copied.
    """
    if isinstance(array, numpy_ndarray) and array.dtype in numpy_floats:
        array = array.round(0).astype(numpy_uint32)
    return array_to_surface(surface, array)