from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def pixels_alpha(surface):
    """pygame.surfarray.pixels_alpha(Surface): return array

    reference pixel alphas into a 2d array

    Create a new 2D array that directly references the alpha values
    (degree of transparency) in a Surface. Any changes to the array will
    affect the pixels in the Surface. This is a fast operation since no
    data is copied.

    This can only work on 32-bit Surfaces with a per-pixel alpha value.

    The Surface this array references will remain locked for the
    lifetime of the array.
    """
    return numpy.array(surface.get_view('A'), copy=False)