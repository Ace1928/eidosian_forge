from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def reset_cooldown() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Resets the cooldown time, if any, to allow the next FedCM call to show
    a dialog even if one was recently dismissed by the user.
    """
    cmd_dict: T_JSON_DICT = {'method': 'FedCm.resetCooldown'}
    json = (yield cmd_dict)