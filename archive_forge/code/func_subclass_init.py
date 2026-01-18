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
def subclass_init(self, cls: type[t.Any]) -> None:
    if isinstance(self._value_trait, TraitType):
        self._value_trait.subclass_init(cls)
    if isinstance(self._key_trait, TraitType):
        self._key_trait.subclass_init(cls)
    if self._per_key_traits is not None:
        for trait in self._per_key_traits.values():
            trait.subclass_init(cls)