from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
def clear_events(service: ServiceName) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears all stored data for the service.

    :param service:
    """
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'BackgroundService.clearEvents', 'params': params}
    json = (yield cmd_dict)