from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def get_target_info(target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, TargetInfo]:
    """
    Returns information about a target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)*
    :returns: 
    """
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.getTargetInfo', 'params': params}
    json = (yield cmd_dict)
    return TargetInfo.from_json(json['targetInfo'])