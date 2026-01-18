from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def discard_console_entries() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Discards collected exceptions and console API calls.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.discardConsoleEntries'}
    json = (yield cmd_dict)