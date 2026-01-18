from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def pixels_blue(surface):
    """pygame.surfarray.pixels_blue(Surface): return array

    Reference pixel blue into a 2d array.

    Create a new 2D array that directly references the blue values
    in a Surface. Any changes to the array will affect the pixels
    in the Surface. This is a fast operation since no data is copied.

    This can only work on 24-bit or 32-bit Surfaces.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    return numpy.array(surface.get_view('B'), copy=False)