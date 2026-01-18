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
def navigate_to_history_entry(entry_id: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Navigates current page to the given history entry.

    :param entry_id: Unique id of the entry to navigate to.
    """
    params: T_JSON_DICT = dict()
    params['entryId'] = entry_id
    cmd_dict: T_JSON_DICT = {'method': 'Page.navigateToHistoryEntry', 'params': params}
    json = (yield cmd_dict)