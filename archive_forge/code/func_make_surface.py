from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def make_surface(array):
    """pygame.surfarray.make_surface (array): return Surface

    Copy an array to a new surface.

    Create a new Surface that best resembles the data and format on the
    array. The array can be 2D or 3D with any sized integer values.
    """
    if isinstance(array, numpy_ndarray) and array.dtype in numpy_floats:
        array = array.round(0).astype(numpy_uint32)
    return pix_make_surface(array)