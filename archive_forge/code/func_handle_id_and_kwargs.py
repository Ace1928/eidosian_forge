from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def handle_id_and_kwargs(obj: Dict[str, Any], root: bool=False) -> Dict[str, Any]:
    """Recursively handles the id and kwargs fields of a dictionary.
            changes the id field to a string "_kind" field that tells WBTraceTree how
            to visualize the run. recursively moves the dictionaries under the kwargs
            key to the top level.
            :param obj: a run dictionary with id and kwargs fields.
            :param root: whether this is the root dictionary or the serialized
                dictionary.
            :return: The modified dictionary.
            """
    if isinstance(obj, dict):
        if ('id' in obj or 'name' in obj) and (not root):
            _kind = obj.get('id')
            if not _kind:
                _kind = [obj.get('name')]
            obj['_kind'] = _kind[-1]
            obj.pop('id', None)
            obj.pop('name', None)
            if 'kwargs' in obj:
                kwargs = obj.pop('kwargs')
                for k, v in kwargs.items():
                    obj[k] = v
        for k, v in obj.items():
            obj[k] = handle_id_and_kwargs(v)
    elif isinstance(obj, list):
        obj = [handle_id_and_kwargs(x) for x in obj]
    return obj