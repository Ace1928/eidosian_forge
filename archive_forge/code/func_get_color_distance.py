from math import sqrt
from functools import lru_cache
from typing import Sequence, Tuple, TYPE_CHECKING
from .color_triplet import ColorTriplet
def get_color_distance(index: int) -> float:
    """Get the distance to a color."""
    red2, green2, blue2 = get_color(index)
    red_mean = (red1 + red2) // 2
    red = red1 - red2
    green = green1 - green2
    blue = blue1 - blue2
    return _sqrt(((512 + red_mean) * red * red >> 8) + 4 * green * green + ((767 - red_mean) * blue * blue >> 8))