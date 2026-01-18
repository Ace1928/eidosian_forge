from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def array3d(surface):
    """pygame.surfarray.array3d(Surface): return array

    copy pixels into a 3d array

    Copy the pixels from a Surface into a 3D array. The bit depth of the
    surface will control the size of the integer values, and will work
    for any type of pixel format.

    This function will temporarily lock the Surface as pixels are copied
    (see the Surface.lock - lock the Surface memory for pixel access
    method).
    """
    width, height = surface.get_size()
    array = numpy.empty((width, height, 3), numpy.uint8)
    surface_to_array(array, surface)
    return array