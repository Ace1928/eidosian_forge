from __future__ import annotations
import logging # isort:skip
from inspect import Parameter, Signature, isclass
from typing import TYPE_CHECKING, Any, Iterable
from ..core import properties as p
from ..core.has_props import HasProps, _default_resolver, abstract
from ..core.property._sphinx import type_link
from ..core.property.validation import without_property_validation
from ..core.serialization import ObjectRefRep, Ref, Serializer
from ..core.types import ID
from ..events import Event
from ..themes import default as default_theme
from ..util.callback_manager import EventCallbackManager, PropertyCallbackManager
from ..util.serialization import make_id
from .docs import html_repr, process_example
from .util import (
def set_select(self, selector: type[Model] | SelectorType, updates: dict[str, Any]) -> None:
    """ Update objects that match a given selector with the specified
        attribute/value updates.

        Args:
            selector (JSON-like) :
            updates (dict) :

        Returns:
            None

        """
    if isclass(selector) and issubclass(selector, Model):
        selector = dict(type=selector)
    for obj in self.select(selector):
        for key, val in updates.items():
            setattr(obj, key, val)