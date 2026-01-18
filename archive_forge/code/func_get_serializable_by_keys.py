import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def get_serializable_by_keys(obj: Any) -> List[SerializableByKey]:
    """Returns all SerializableByKeys contained by obj.

    Objects are ordered such that nested objects appear before the object they
    are nested inside. This is required to ensure SerializableByKeys are only
    fully defined once in serialization.
    """
    result = []
    if isinstance(obj, SerializableByKey):
        result.append(obj)
    json_dict = getattr(obj, '_json_dict_', lambda: None)()
    if isinstance(json_dict, Dict):
        for v in json_dict.values():
            result = get_serializable_by_keys(v) + result
    if result:
        return result
    if isinstance(obj, Dict):
        return [sbk for pair in obj.items() for sbk in get_serializable_by_keys(pair)]
    if hasattr(obj, '__iter__') and (not isinstance(obj, str)):
        return [sbk for v in obj for sbk in get_serializable_by_keys(v)]
    return []