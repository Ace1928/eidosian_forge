from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def detach_from_target(session_id: typing.Optional[SessionID]=None, target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Detaches session with given id.

    :param session_id: *(Optional)* Session to detach.
    :param target_id: *(Optional)* Deprecated.
    """
    params: T_JSON_DICT = dict()
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.detachFromTarget', 'params': params}
    json = (yield cmd_dict)