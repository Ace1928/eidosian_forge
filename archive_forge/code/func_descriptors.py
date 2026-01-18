from __future__ import annotations
import logging # isort:skip
import difflib
from typing import (
from weakref import WeakSet
from ..settings import settings
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import (
from .types import ID
@classmethod
@lru_cache(None)
def descriptors(cls) -> list[PropertyDescriptor[Any]]:
    """ List of property descriptors in the order of definition. """
    return [cls.lookup(name) for name, _ in cls.properties(_with_props=True).items()]