from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
def deliver_push_message(origin: str, registration_id: RegistrationID, data: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param origin:
    :param registration_id:
    :param data:
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['data'] = data
    cmd_dict: T_JSON_DICT = {'method': 'ServiceWorker.deliverPushMessage', 'params': params}
    json = (yield cmd_dict)