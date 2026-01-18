from __future__ import annotations
import logging # isort:skip
from copy import copy
from typing import TypeVar
from ..has_props import HasProps
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
def make_descriptors(self, _base_name: str) -> list[PropertyDescriptor[T]]:
    descriptors = []
    for descriptor in self.delegate.descriptors():
        prop = copy(descriptor.property)
        prop.__doc__ = self.help.format(prop=descriptor.name.replace('_', ' '))
        descriptors += prop.make_descriptors(self.prefix + descriptor.name)
    return descriptors