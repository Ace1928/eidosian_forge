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
def set_transform_rotates_text(self, t):
    """
        Whether rotations of the transform affect the text direction.

        Parameters
        ----------
        t : bool
        """
    self._transform_rotates_text = t
    self.stale = True