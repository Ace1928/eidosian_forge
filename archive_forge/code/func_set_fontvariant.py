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
def set_fontvariant(self, variant):
    """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}

        See Also
        --------
        .font_manager.FontProperties.set_variant
        """
    self._fontproperties.set_variant(variant)
    self.stale = True