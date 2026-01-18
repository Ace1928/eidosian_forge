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
def stop_screencast() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Stops sending each frame in the ``screencastFrame``.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.stopScreencast'}
    json = (yield cmd_dict)