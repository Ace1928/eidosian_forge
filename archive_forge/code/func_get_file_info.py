from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_file_info(object_id: runtime.RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns file information for the given
    File wrapper.

    **EXPERIMENTAL**

    :param object_id: JavaScript object id of the node wrapper.
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getFileInfo', 'params': params}
    json = (yield cmd_dict)
    return str(json['path'])