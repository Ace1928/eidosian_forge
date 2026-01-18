from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_timezone_override(timezone_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides default host system timezone with the specified one.

    :param timezone_id: The timezone identifier. If empty, disables the override and restores default host system timezone.
    """
    params: T_JSON_DICT = dict()
    params['timezoneId'] = timezone_id
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setTimezoneOverride', 'params': params}
    json = (yield cmd_dict)