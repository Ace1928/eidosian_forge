from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def select_prompt(id_: RequestId, device_id: DeviceId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Select a device in response to a DeviceAccess.deviceRequestPrompted event.

    :param id_:
    :param device_id:
    """
    params: T_JSON_DICT = dict()
    params['id'] = id_.to_json()
    params['deviceId'] = device_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DeviceAccess.selectPrompt', 'params': params}
    json = (yield cmd_dict)