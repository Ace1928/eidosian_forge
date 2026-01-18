import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def auto_set_font_size(self, value=True):
    """Automatically set font size."""
    self._autoFontsize = value
    self.stale = True