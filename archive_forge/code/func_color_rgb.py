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
def color_rgb(self, red, green, blue):
    """
        Provides callable formatting string to set foreground color to the specified RGB color.

        :arg int red: RGB value of Red.
        :arg int green: RGB value of Green.
        :arg int blue: RGB value of Blue.
        :rtype: FormattingString
        :returns: Callable string that sets the foreground color

        If the terminal does not support RGB color, the nearest supported
        color will be determined using :py:attr:`color_distance_algorithm`.
        """
    if self.number_of_colors == 1 << 24:
        fmt_attr = u'\x1b[38;2;{0};{1};{2}m'.format(red, green, blue)
        return FormattingString(fmt_attr, self.normal)
    color_idx = self.rgb_downconvert(red, green, blue)
    return FormattingString(self._foreground_color(color_idx), self.normal)