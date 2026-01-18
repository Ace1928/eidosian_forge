from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def pixels3d(surface):
    """pygame.surfarray.pixels3d(Surface): return array

    reference pixels into a 3d array

    Create a new 3D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    This will only work on Surfaces that have 24-bit or 32-bit
    formats. Lower pixel formats cannot be referenced.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    return numpy_array(surface.get_view('3'), copy=False)