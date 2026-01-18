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
def js_on_event(self, event: str | type[Event], *callbacks: JSEventCallback) -> None:
    if isinstance(event, str):
        event_name = Event.cls_for(event).event_name
    elif isinstance(event, type) and issubclass(event, Event):
        event_name = event.event_name
    else:
        raise ValueError(f'expected string event name or event class, got {event}')
    all_callbacks = list(self.js_event_callbacks.get(event_name, []))
    for callback in callbacks:
        if callback not in all_callbacks:
            all_callbacks.append(callback)
    self.js_event_callbacks[event_name] = all_callbacks