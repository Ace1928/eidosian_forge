from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_heap_object_id(object_id: runtime.RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, HeapSnapshotObjectId]:
    """
    :param object_id: Identifier of the object to get heap object id for.
    :returns: Id of the heap snapshot object corresponding to the passed remote object id.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.getHeapObjectId', 'params': params}
    json = (yield cmd_dict)
    return HeapSnapshotObjectId.from_json(json['heapSnapshotObjectId'])