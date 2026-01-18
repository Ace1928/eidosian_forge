import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def get_required_width(self, renderer):
    """Return the minimal required width for the cell."""
    l, b, w, h = self.get_text_bounds(renderer)
    return w * (1.0 + 2.0 * self.PAD)