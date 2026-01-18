from __future__ import unicode_literals
from six.moves import zip_longest, range
from prompt_toolkit.filters import HasCompletions, IsDone, Condition, to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .containers import Window, HSplit, ConditionalContainer, ScrollOffsets
from .controls import UIControl, UIContent
from .dimension import LayoutDimension
from .margins import ScrollbarMargin
from .screen import Point, Char
import math
def grouper(n, iterable, fillvalue=None):
    """ grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)