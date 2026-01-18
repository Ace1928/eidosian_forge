import sys
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from ..._utils import logger_warning
from .. import LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS
from ._font import Font
from ._text_state_manager import TextStateManager
from ._text_state_params import TextStateParams
def bt_group(tj_op: TextStateParams, rendered_text: str, dispaced_tx: float) -> BTGroup:
    """
    BTGroup constructed from a TextStateParams instance, rendered text, and
    displaced tx value.

    Args:
        tj_op (TextStateParams): TextStateParams instance
        rendered_text (str): rendered text
        dispaced_tx (float): x coordinate of last character in BTGroup
    """
    return BTGroup(tx=tj_op.tx, ty=tj_op.ty, font_size=tj_op.font_size, font_height=tj_op.font_height, text=rendered_text, displaced_tx=dispaced_tx, flip_sort=-1 if tj_op.flip_vertical else 1)