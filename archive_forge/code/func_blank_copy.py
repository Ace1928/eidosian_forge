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
def blank_copy(self, plain: str='') -> 'Text':
    """Return a new Text instance with copied meta data (but not the string or spans)."""
    copy_self = Text(plain, style=self.style, justify=self.justify, overflow=self.overflow, no_wrap=self.no_wrap, end=self.end, tab_size=self.tab_size)
    return copy_self