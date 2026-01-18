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
def rstrip_end(self, size: int) -> None:
    """Remove whitespace beyond a certain width at the end of the text.

        Args:
            size (int): The desired size of the text.
        """
    text_length = len(self)
    if text_length > size:
        excess = text_length - size
        whitespace_match = _re_whitespace.search(self.plain)
        if whitespace_match is not None:
            whitespace_count = len(whitespace_match.group(0))
            self.right_crop(min(whitespace_count, excess))