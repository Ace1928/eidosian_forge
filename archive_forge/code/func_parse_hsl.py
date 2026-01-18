import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def parse_hsl(h: str, h_units: str, sat: str, light: str, alpha: Optional[float]=None) -> RGBA:
    """
    Parse raw hue, saturation, lightness and alpha values and convert to RGBA.
    """
    s_value, l_value = (parse_color_value(sat, 100), parse_color_value(light, 100))
    h_value = float(h)
    if h_units in {None, 'deg'}:
        h_value = h_value % 360 / 360
    elif h_units == 'rad':
        h_value = h_value % rads / rads
    else:
        h_value = h_value % 1
    r, g, b = hls_to_rgb(h_value, l_value, s_value)
    return RGBA(r, g, b, alpha)