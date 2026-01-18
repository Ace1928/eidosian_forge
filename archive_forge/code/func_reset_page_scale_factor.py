from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def reset_page_scale_factor() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests that page scale factor is reset to initial values.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.resetPageScaleFactor'}
    json = (yield cmd_dict)