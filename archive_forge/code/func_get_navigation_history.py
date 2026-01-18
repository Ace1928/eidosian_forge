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
def get_navigation_history() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[int, typing.List[NavigationEntry]]]:
    """
    Returns navigation history for the current page.

    :returns: A tuple with the following items:

        0. **currentIndex** - Index of the current navigation history entry.
        1. **entries** - Array of navigation history entries.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getNavigationHistory'}
    json = (yield cmd_dict)
    return (int(json['currentIndex']), [NavigationEntry.from_json(i) for i in json['entries']])