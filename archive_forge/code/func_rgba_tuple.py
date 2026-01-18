import numpy as np
from bokeh.core.properties import (
from ...core.options import abbreviated_exception
from ...core.util import arraylike_types
from ...util.transform import dim
from ..util import COLOR_ALIASES, RGB_HEX_REGEX, rgb2hex
def rgba_tuple(rgba):
    """
    Ensures RGB(A) tuples in the range 0-1 are scaled to 0-255.
    """
    if isinstance(rgba, tuple):
        return tuple((int(c * 255) if i < 3 else c for i, c in enumerate(rgba)))
    else:
        return COLOR_ALIASES.get(rgba, rgba)