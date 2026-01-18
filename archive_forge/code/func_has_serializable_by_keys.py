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
def has_serializable_by_keys(obj: Any) -> bool:
    """Returns true if obj contains one or more SerializableByKey objects."""
    if isinstance(obj, SerializableByKey):
        return True
    json_dict = getattr(obj, '_json_dict_', lambda: None)()
    if isinstance(json_dict, Dict):
        return any((has_serializable_by_keys(v) for v in json_dict.values()))
    if isinstance(obj, Dict):
        return any((has_serializable_by_keys(elem) for pair in obj.items() for elem in pair))
    if hasattr(obj, '__iter__') and (not isinstance(obj, str)):
        try:
            return any((has_serializable_by_keys(elem) for elem in obj))
        except TypeError:
            return False
    return False