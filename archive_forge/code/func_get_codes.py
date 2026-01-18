import sys
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Union
from .segment import ControlCode, ControlType, Segment
def get_codes() -> Iterable[ControlCode]:
    control = ControlType
    if x:
        yield (control.CURSOR_FORWARD if x > 0 else control.CURSOR_BACKWARD, abs(x))
    if y:
        yield (control.CURSOR_DOWN if y > 0 else control.CURSOR_UP, abs(y))