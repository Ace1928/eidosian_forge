from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def modify_serialized_iterative(self, runs: List[Dict[str, Any]], exact_keys: Tuple[str, ...]=(), partial_keys: Tuple[str, ...]=()) -> List[Dict[str, Any]]:
    """Utility to modify the serialized field of a list of runs dictionaries.
        removes any keys that match the exact_keys and any keys that contain any of the
        partial_keys.
        recursively moves the dictionaries under the kwargs key to the top level.
        changes the "id" field to a string "_kind" field that tells WBTraceTree how to
        visualize the run. promotes the "serialized" field to the top level.

        :param runs: The list of runs to modify.
        :param exact_keys: A tuple of keys to remove from the serialized field.
        :param partial_keys: A tuple of partial keys to remove from the serialized
            field.
        :return: The modified list of runs.
        """

    def remove_exact_and_partial_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively removes exact and partial keys from a dictionary.
            :param obj: The dictionary to remove keys from.
            :return: The modified dictionary.
            """
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k not in exact_keys and (not any((partial in k for partial in partial_keys)))}
            for k, v in obj.items():
                obj[k] = remove_exact_and_partial_keys(v)
        elif isinstance(obj, list):
            obj = [remove_exact_and_partial_keys(x) for x in obj]
        return obj

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

    def transform_serialized(serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms the serialized field of a run dictionary to be compatible
                with WBTraceTree.
            :param serialized: The serialized field of a run dictionary.
            :return: The transformed serialized field.
            """
        serialized = handle_id_and_kwargs(serialized, root=True)
        serialized = remove_exact_and_partial_keys(serialized)
        return serialized

    def transform_run(run: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms a run dictionary to be compatible with WBTraceTree.
            :param run: The run dictionary to transform.
            :return: The transformed run dictionary.
            """
        transformed_dict = transform_serialized(run)
        serialized = transformed_dict.pop('serialized')
        for k, v in serialized.items():
            transformed_dict[k] = v
        _kind = transformed_dict.get('_kind', None)
        name = transformed_dict.pop('name', None)
        exec_ord = transformed_dict.pop('execution_order', None)
        if not name:
            name = _kind
        output_dict = {f'{exec_ord}_{name}': transformed_dict}
        return output_dict
    return list(map(transform_run, runs))