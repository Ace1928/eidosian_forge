import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def extract_type(t):
    origin = typing.get_origin(t)
    if origin in [typing.Union, list]:
        type_arg = typing.get_args(t)[0]
        return extract_type(type_arg)
    elif origin == dict:
        value_type_arg = typing.get_args(t)[1]
        return extract_type(value_type_arg)
    elif is_dataclass(t) or issubclass_safe(t, (Enum, date, datetime)):
        return t
    return None