from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def simulate_pressure_notification(level: PressureLevel) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Simulate a memory pressure notification in all processes.

    :param level: Memory pressure level of the notification.
    """
    params: T_JSON_DICT = dict()
    params['level'] = level.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Memory.simulatePressureNotification', 'params': params}
    json = (yield cmd_dict)