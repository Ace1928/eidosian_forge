import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def expr_builder_from(t: type, depth=0):
    return expr_builder(t, depth, direction=_FROM)