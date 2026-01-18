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
def bring_to_front() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Brings page to front (activates tab).
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.bringToFront'}
    json = (yield cmd_dict)