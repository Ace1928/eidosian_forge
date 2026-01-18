from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_pressure_notifications_suppressed(suppressed: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable/disable suppressing memory pressure notifications in all processes.

    :param suppressed: If true, memory pressure notifications will be suppressed.
    """
    params: T_JSON_DICT = dict()
    params['suppressed'] = suppressed
    cmd_dict: T_JSON_DICT = {'method': 'Memory.setPressureNotificationsSuppressed', 'params': params}
    json = (yield cmd_dict)