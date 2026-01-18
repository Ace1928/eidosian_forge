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
def detect_indentation(self) -> int:
    """Auto-detect indentation of code.

        Returns:
            int: Number of spaces used to indent code.
        """
    _indentations = {len(match.group(1)) for match in re.finditer('^( *)(.*)$', self.plain, flags=re.MULTILINE)}
    try:
        indentation = reduce(gcd, [indent for indent in _indentations if not indent % 2]) or 1
    except TypeError:
        indentation = 1
    return indentation