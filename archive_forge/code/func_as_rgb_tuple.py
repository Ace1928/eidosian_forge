import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_rgb_tuple(self, *, alpha: Optional[bool]=None) -> ColorTuple:
    """
        Color as an RGB or RGBA tuple; red, green and blue are in the range 0 to 255, alpha if included is
        in the range 0 to 1.

        :param alpha: whether to include the alpha channel, options are
          None - (default) include alpha only if it's set (e.g. not None)
          True - always include alpha,
          False - always omit alpha,
        """
    r, g, b = (float_to_255(c) for c in self._rgba[:3])
    if alpha is None:
        if self._rgba.alpha is None:
            return (r, g, b)
        else:
            return (r, g, b, self._alpha_float())
    elif alpha:
        return (r, g, b, self._alpha_float())
    else:
        return (r, g, b)