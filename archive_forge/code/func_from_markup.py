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
@classmethod
def from_markup(cls, text: str, *, style: Union[str, Style]='', emoji: bool=True, emoji_variant: Optional[EmojiVariant]=None, justify: Optional['JustifyMethod']=None, overflow: Optional['OverflowMethod']=None, end: str='\n') -> 'Text':
    """Create Text instance from markup.

        Args:
            text (str): A string containing console markup.
            emoji (bool, optional): Also render emoji code. Defaults to True.
            justify (str, optional): Justify method: "left", "center", "full", "right". Defaults to None.
            overflow (str, optional): Overflow method: "crop", "fold", "ellipsis". Defaults to None.
            end (str, optional): Character to end text with. Defaults to "\\\\n".

        Returns:
            Text: A Text instance with markup rendered.
        """
    from .markup import render
    rendered_text = render(text, style, emoji=emoji, emoji_variant=emoji_variant)
    rendered_text.justify = justify
    rendered_text.overflow = overflow
    rendered_text.end = end
    return rendered_text