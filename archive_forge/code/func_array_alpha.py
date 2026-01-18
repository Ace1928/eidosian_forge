from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def array_alpha(surface):
    """pygame.surfarray.array_alpha(Surface): return array

    copy pixel alphas into a 2d array

    Copy the pixel alpha values (degree of transparency) from a Surface
    into a 2D array. This will work for any type of Surface
    format. Surfaces without a pixel alpha will return an array with all
    opaque values.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    size = surface.get_size()
    array = numpy.empty(size, numpy.uint8)
    surface_to_array(array, surface, 'A')
    return array