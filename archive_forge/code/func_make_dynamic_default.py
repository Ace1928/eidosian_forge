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
def make_dynamic_default(self) -> T | None:
    if self.default_args is None and self.default_kwargs is None:
        return None
    assert self.klass is not None
    return self.klass(*(self.default_args or ()), **self.default_kwargs or {})