from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def array2d(surface):
    """pygame.surfarray.array2d(Surface): return array

    copy pixels into a 2d array

    Copy the pixels from a Surface into a 2D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    bpp = surface.get_bytesize()
    try:
        dtype = (numpy.uint8, numpy.uint16, numpy.int32, numpy.int32)[bpp - 1]
    except IndexError:
        raise ValueError(f'unsupported bit depth {bpp * 8} for 2D array')
    size = surface.get_size()
    array = numpy.empty(size, dtype)
    surface_to_array(array, surface)
    return array