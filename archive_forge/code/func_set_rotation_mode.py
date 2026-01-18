import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def set_rotation_mode(self, m):
    """
        Set text rotation mode.

        Parameters
        ----------
        m : {None, 'default', 'anchor'}
            If ``"default"``, the text will be first rotated, then aligned according
            to their horizontal and vertical alignments.  If ``"anchor"``, then
            alignment occurs before rotation. Passing ``None`` will set the rotation
            mode to ``"default"``.
        """
    if m is None:
        m = 'default'
    else:
        _api.check_in_list(('anchor', 'default'), rotation_mode=m)
    self._rotation_mode = m
    self.stale = True