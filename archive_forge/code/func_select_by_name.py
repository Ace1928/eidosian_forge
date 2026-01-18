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
def select_by_name(self, value: str, default: t.Any=Undefined) -> t.Any:
    """Selects enum-value by using its name or scoped-name."""
    assert isinstance(value, str)
    if value.startswith(self.name_prefix):
        value = value.replace(self.name_prefix, '', 1)
    return self.enum_class.__members__.get(value, default)