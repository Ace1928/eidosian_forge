from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def attach_to_target(target_id: TargetID, flatten: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SessionID]:
    """
    Attaches to the target with given id.

    :param target_id:
    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :returns: Id assigned to the session.
    """
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if flatten is not None:
        params['flatten'] = flatten
    cmd_dict: T_JSON_DICT = {'method': 'Target.attachToTarget', 'params': params}
    json = (yield cmd_dict)
    return SessionID.from_json(json['sessionId'])