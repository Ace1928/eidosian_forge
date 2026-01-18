from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_object_by_heap_object_id(object_id: HeapSnapshotObjectId, object_group: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, runtime.RemoteObject]:
    """
    :param object_id:
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :returns: Evaluation result.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.getObjectByHeapObjectId', 'params': params}
    json = (yield cmd_dict)
    return runtime.RemoteObject.from_json(json['result'])