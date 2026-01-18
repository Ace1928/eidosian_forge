import re
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from typing import (
from ._loop import loop_last
from ._pick import pick_bool
from ._wrap import divide_line
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .containers import Lines
from .control import strip_control_codes
from .emoji import EmojiVariant
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
def extend_style(self, spaces: int) -> None:
    """Extend the Text given number of spaces where the spaces have the same style as the last character.

        Args:
            spaces (int): Number of spaces to add to the Text.
        """
    if spaces <= 0:
        return
    spans = self.spans
    new_spaces = ' ' * spaces
    if spans:
        end_offset = len(self)
        self._spans[:] = [span.extend(spaces) if span.end >= end_offset else span for span in spans]
        self._text.append(new_spaces)
        self._length += spaces
    else:
        self.plain += new_spaces