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
def init_default_value(self, obj: t.Any) -> G | None:
    """DEPRECATED: Set the static default value for the trait type."""
    warn('init_default_value is deprecated in traitlets 4.0, and may be removed in the future', DeprecationWarning, stacklevel=2)
    value = self._validate(obj, self.default_value)
    obj._trait_values[self.name] = value
    return value