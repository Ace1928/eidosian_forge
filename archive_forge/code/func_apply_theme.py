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
def apply_theme(self, property_values: dict[str, Any]) -> None:
    """ Apply a set of theme values which will be used rather than
        defaults, but will not override application-set values.

        The passed-in dictionary may be kept around as-is and shared with
        other instances to save memory (so neither the caller nor the
        |HasProps| instance should modify it).

        Args:
            property_values (dict) : theme values to use in place of defaults

        Returns:
            None

        """
    old_dict = self.themed_values()
    if old_dict is property_values:
        return
    removed: set[str] = set()
    if old_dict is not None:
        removed.update(set(old_dict.keys()))
    added = set(property_values.keys())
    old_values: dict[str, Any] = {}
    for k in added.union(removed):
        old_values[k] = getattr(self, k)
    if len(property_values) > 0:
        setattr(self, '__themed_values__', property_values)
    elif hasattr(self, '__themed_values__'):
        delattr(self, '__themed_values__')
    for k, v in old_values.items():
        if k in self._unstable_themed_values:
            del self._unstable_themed_values[k]
    for k, v in old_values.items():
        descriptor = self.lookup(k)
        if isinstance(descriptor, PropertyDescriptor):
            descriptor.trigger_if_changed(self, v)