import sys
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Union
from .segment import ControlCode, ControlType, Segment
def escape_control_codes(text: str, _translate_table: Dict[int, str]=CONTROL_ESCAPE) -> str:
    """Replace control codes with their "escaped" equivalent in the given text.
    (e.g. "\x08" becomes "\\b")

    Args:
        text (str): A string possibly containing control codes.

    Returns:
        str: String with control codes replaced with their escaped version.
    """
    return text.translate(_translate_table)