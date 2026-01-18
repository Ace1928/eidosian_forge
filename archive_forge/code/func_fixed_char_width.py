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
def fixed_char_width(bt_groups: List[BTGroup], scale_weight: float=1.25) -> float:
    """
    Calculate average character width weighted by the length of the rendered
    text in each sample for conversion to fixed-width layout.

    Args:
        bt_groups (List[BTGroup]): List of dicts of text rendered by each
            BT operator

    Returns:
        float: fixed character width
    """
    char_widths = []
    for _bt in bt_groups:
        _len = len(_bt['text']) * scale_weight
        char_widths.append(((_bt['displaced_tx'] - _bt['tx']) / _len, _len))
    return sum((_w * _l for _w, _l in char_widths)) / sum((_l for _, _l in char_widths))