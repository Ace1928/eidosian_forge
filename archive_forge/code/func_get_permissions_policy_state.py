from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_permissions_policy_state(frame_id: FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[PermissionsPolicyFeatureState]]:
    """
    Get Permissions Policy state on given frame.

    **EXPERIMENTAL**

    :param frame_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.getPermissionsPolicyState', 'params': params}
    json = (yield cmd_dict)
    return [PermissionsPolicyFeatureState.from_json(i) for i in json['states']]