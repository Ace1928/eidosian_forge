from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def hide_highlight() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Hides any highlight.
    """
    cmd_dict: T_JSON_DICT = {'method': 'DOM.hideHighlight'}
    json = (yield cmd_dict)