from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def array_colorkey(surface):
    """pygame.surfarray.array_colorkey(Surface): return array

    copy the colorkey values into a 2d array

    Create a new array with the colorkey transparency value from each
    pixel. If the pixel matches the colorkey it will be fully
    transparent; otherwise it will be fully opaque.

    This will work on any type of Surface format. If the image has no
    colorkey a solid opaque array will be returned.

    This function will temporarily lock the Surface as pixels are
    copied.
    """
    size = surface.get_size()
    array = numpy.empty(size, numpy.uint8)
    surface_to_array(array, surface, 'C')
    return array