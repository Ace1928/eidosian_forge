import os
import re
import sys
import time
import codecs
import locale
import select
import struct
import platform
import warnings
import functools
import contextlib
import collections
from .color import COLOR_DISTANCE_ALGORITHMS
from .keyboard import (_time_left,
from .sequences import Termcap, Sequence, SequenceTextWrapper
from .colorspace import RGB_256TABLE
from .formatters import (COLORS,
from ._capabilities import CAPABILITY_DATABASE, CAPABILITIES_ADDITIVES, CAPABILITIES_RAW_MIXIN
def rgb_downconvert(self, red, green, blue):
    """
        Translate an RGB color to a color code of the terminal's color depth.

        :arg int red: RGB value of Red (0-255).
        :arg int green: RGB value of Green (0-255).
        :arg int blue: RGB value of Blue (0-255).
        :rtype: int
        :returns: Color code of downconverted RGB color
        """
    fn_distance = COLOR_DISTANCE_ALGORITHMS[self.color_distance_algorithm]
    color_idx = 7
    shortest_distance = None
    for cmp_depth, cmp_rgb in enumerate(RGB_256TABLE):
        cmp_distance = fn_distance(cmp_rgb, (red, green, blue))
        if shortest_distance is None or cmp_distance < shortest_distance:
            shortest_distance = cmp_distance
            color_idx = cmp_depth
        if cmp_depth >= self.number_of_colors:
            break
    return color_idx