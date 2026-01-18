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
def pad_right(self, count: int, character: str=' ') -> None:
    """Pad the right with a given character.

        Args:
            count (int): Number of characters to pad.
            character (str, optional): Character to pad with. Defaults to " ".
        """
    assert len(character) == 1, 'Character must be a string of length 1'
    if count:
        self.plain = f'{self.plain}{character * count}'