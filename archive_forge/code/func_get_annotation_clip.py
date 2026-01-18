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
def get_annotation_clip(self):
    """
        Return the annotation's clipping behavior.

        See `set_annotation_clip` for the meaning of return values.
        """
    return self._annotation_clip