from __future__ import annotations
import re
from . import Image, _imagingmorph
def get_on_pixels(self, image):
    """Get a list of all turned on pixels in a binary image

        Returns a list of tuples of (x,y) coordinates
        of all matching pixels. See :ref:`coordinate-system`."""
    if image.mode != 'L':
        msg = 'Image mode must be L'
        raise ValueError(msg)
    return _imagingmorph.get_on_pixels(image.im.id)