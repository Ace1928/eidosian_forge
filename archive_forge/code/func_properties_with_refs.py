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
def properties_with_refs(cls) -> dict[str, Property[Any]]:
    """ Collect the names of all properties on this class that also have
        references.

        This method *always* traverses the class hierarchy and includes
        properties defined on any parent classes.

        Returns:
            set[str] : names of properties that have references

        """
    return {k: v for k, v in cls.properties(_with_props=True).items() if v.has_ref}