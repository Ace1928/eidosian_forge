from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def cancel_dragging() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Cancels any active dragging in the page.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Input.cancelDragging'}
    json = (yield cmd_dict)