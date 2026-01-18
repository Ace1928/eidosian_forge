import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def parse_color_value(value: Union[int, str], max_val: int=255) -> float:
    """
    Parse a value checking it's a valid int in the range 0 to max_val and divide by max_val to give a number
    in the range 0 to 1
    """
    try:
        color = float(value)
    except ValueError:
        raise ColorError(reason='color values must be a valid number')
    if 0 <= color <= max_val:
        return color / max_val
    else:
        raise ColorError(reason=f'color values must be in the range 0 to {max_val}')