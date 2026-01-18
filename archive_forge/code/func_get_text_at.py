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
def get_text_at(offset: int) -> 'Text':
    _Span = Span
    text = Text(self.plain[offset], spans=[_Span(0, 1, style) for start, end, style in self._spans if end > offset >= start], end='')
    return text