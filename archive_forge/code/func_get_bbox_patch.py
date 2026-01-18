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
def get_bbox_patch(self):
    """
        Return the bbox Patch, or None if the `.patches.FancyBboxPatch`
        is not made.
        """
    return self._bbox_patch