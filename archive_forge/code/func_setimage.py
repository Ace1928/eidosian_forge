from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def setimage(self, im, extents=None):
    """
        Called from ImageFile to set the core output image for the codec

        :param im: A core image object
        :param extents: a 4 tuple of (x0, y0, x1, y1) defining the rectangle
            for this tile
        :returns: None
        """
    self.im = im
    if extents:
        x0, y0, x1, y1 = extents
    else:
        x0, y0, x1, y1 = (0, 0, 0, 0)
    if x0 == 0 and x1 == 0:
        self.state.xsize, self.state.ysize = self.im.size
    else:
        self.state.xoff = x0
        self.state.yoff = y0
        self.state.xsize = x1 - x0
        self.state.ysize = y1 - y0
    if self.state.xsize <= 0 or self.state.ysize <= 0:
        msg = 'Size cannot be negative'
        raise ValueError(msg)
    if self.state.xsize + self.state.xoff > self.im.size[0] or self.state.ysize + self.state.yoff > self.im.size[1]:
        msg = 'Tile cannot extend outside image'
        raise ValueError(msg)