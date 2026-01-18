from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def can_emulate() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    """
    Tells whether emulation is supported.

    :returns: True if emulation is supported.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.canEmulate'}
    json = (yield cmd_dict)
    return bool(json['result'])