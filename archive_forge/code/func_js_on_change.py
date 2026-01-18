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
def js_on_change(self, event: str, *callbacks: JSChangeCallback) -> None:
    """ Attach a :class:`~bokeh.models.CustomJS` callback to an arbitrary
        BokehJS model event.

        On the BokehJS side, change events for model properties have the
        form ``"change:property_name"``. As a convenience, if the event name
        passed to this method is also the name of a property on the model,
        then it will be prefixed with ``"change:"`` automatically:

        .. code:: python

            # these two are equivalent
            source.js_on_change('data', callback)
            source.js_on_change('change:data', callback)

        However, there are other kinds of events that can be useful to respond
        to, in addition to property change events. For example to run a
        callback whenever data is streamed to a ``ColumnDataSource``, use the
        ``"stream"`` event on the source:

        .. code:: python

            source.js_on_change('streaming', callback)

        """
    if len(callbacks) == 0:
        raise ValueError('js_on_change takes an event name and one or more callbacks, got only one parameter')
    from bokeh.models.callbacks import CustomCode
    if not all((isinstance(x, CustomCode) for x in callbacks)):
        raise ValueError('not all callback values are CustomCode instances')
    descriptor = self.lookup(event, raises=False)
    if descriptor is not None:
        event = f'change:{descriptor.name}'
    old = {k: [cb for cb in cbs] for k, cbs in self.js_property_callbacks.items()}
    if event not in self.js_property_callbacks:
        self.js_property_callbacks[event] = []
    for callback in callbacks:
        if callback in self.js_property_callbacks[event]:
            continue
        self.js_property_callbacks[event].append(callback)
    self.trigger('js_property_callbacks', old, self.js_property_callbacks)