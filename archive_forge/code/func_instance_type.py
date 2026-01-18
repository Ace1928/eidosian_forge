from __future__ import annotations
import logging # isort:skip
import types
from importlib import import_module
from typing import (
from ..has_props import HasProps
from ..serialization import Serializable
from ._sphinx import model_link, property_link, register_type_link
from .bases import Init, Property
from .singletons import Undefined
@property
def instance_type(self) -> type[T]:
    instance_type: type[Serializable]
    if isinstance(self._instance_type, type):
        instance_type = self._instance_type
    elif isinstance(self._instance_type, str):
        module, name = self._instance_type.rsplit('.', 1)
        instance_type = getattr(import_module(module, 'bokeh'), name)
        self._assert_type(instance_type)
        self._instance_type = instance_type
    else:
        instance_type = self._instance_type()
        self._assert_type(instance_type)
        self._instance_type = instance_type
    return instance_type