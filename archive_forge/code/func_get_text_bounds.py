import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def get_text_bounds(self, renderer):
    """
        Return the text bounds as *(x, y, width, height)* in table coordinates.
        """
    return self._text.get_window_extent(renderer).transformed(self.get_data_transform().inverted()).bounds