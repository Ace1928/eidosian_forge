from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
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