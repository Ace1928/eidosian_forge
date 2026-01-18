from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def reset_navigation_history() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Resets navigation history for the current page.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.resetNavigationHistory'}
    json = (yield cmd_dict)