import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_rgb(self) -> str:
    """
        Color as an rgb(<r>, <g>, <b>) or rgba(<r>, <g>, <b>, <a>) string.
        """
    if self._rgba.alpha is None:
        return f'rgb({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)})'
    else:
        return f'rgba({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)}, {round(self._alpha_float(), 2)})'