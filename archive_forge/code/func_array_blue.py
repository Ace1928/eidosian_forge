from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def array_blue(surface):
    """pygame.surfarray.array_blue(Surface): return array

    copy pixel blue into a 2d array

    Copy the pixel blue values from a Surface into a 2D array. This will work
    for any type of Surface format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    size = surface.get_size()
    array = numpy.empty(size, numpy.uint8)
    surface_to_array(array, surface, 'B')
    return array