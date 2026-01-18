from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
def hue_to_rgb(lum1: float, lum2: float, hue: float) -> float:
    if hue < 0.0:
        hue += 1
    if hue > 1.0:
        hue -= 1
    if hue < 1.0 / 6.0:
        return lum1 + (lum2 - lum1) * 6.0 * hue
    if hue < 1.0 / 2.0:
        return lum2
    if hue < 2.0 / 3.0:
        return lum1 + (lum2 - lum1) * (2.0 / 3.0 - hue) * 6.0
    return lum1