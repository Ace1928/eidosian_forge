from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def add_inspected_heap_object(heap_object_id: HeapSnapshotObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables console to refer to the node with given id via $x (see Command Line API for more details
    $x functions).

    :param heap_object_id: Heap snapshot object id to be accessible by means of $x command line API.
    """
    params: T_JSON_DICT = dict()
    params['heapObjectId'] = heap_object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.addInspectedHeapObject', 'params': params}
    json = (yield cmd_dict)