from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
def select_by_number(self, value: int, default: t.Any=Undefined) -> t.Any:
    """Selects enum-value by using its number-constant."""
    assert isinstance(value, int)
    enum_members = self.enum_class.__members__
    for enum_item in enum_members.values():
        if enum_item.value == value:
            return enum_item
    return default