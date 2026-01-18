import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def ints_to_rgba(r: Union[int, str], g: Union[int, str], b: Union[int, str], alpha: Optional[float]) -> RGBA:
    return RGBA(parse_color_value(r), parse_color_value(g), parse_color_value(b), parse_float_alpha(alpha))