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
def js_link(self, attr: str, other: Model, other_attr: str, attr_selector: int | str | None=None) -> None:
    """ Link two Bokeh model properties using JavaScript.

        This is a convenience method that simplifies adding a
        :class:`~bokeh.models.CustomJS` callback to update one Bokeh model
        property whenever another changes value.

        Args:

            attr (str) :
                The name of a Bokeh property on this model

            other (Model):
                A Bokeh model to link to self.attr

            other_attr (str) :
                The property on ``other`` to link together

            attr_selector (Union[int, str]) :
                The index to link an item in a subscriptable ``attr``

        Added in version 1.1

        Raises:

            ValueError

        Examples:

            This code with ``js_link``:

            .. code :: python

                select.js_link('value', plot, 'sizing_mode')

            is equivalent to the following:

            .. code:: python

                from bokeh.models import CustomJS
                select.js_on_change('value',
                    CustomJS(args=dict(other=plot),
                             code="other.sizing_mode = this.value"
                    )
                )

            Additionally, to use attr_selector to attach the left side of a range slider to a plot's x_range:

            .. code :: python

                range_slider.js_link('value', plot.x_range, 'start', attr_selector=0)

            which is equivalent to:

            .. code :: python

                from bokeh.models import CustomJS
                range_slider.js_on_change('value',
                    CustomJS(args=dict(other=plot.x_range),
                             code="other.start = this.value[0]"
                    )
                )

        """
    descriptor = self.lookup(attr, raises=False)
    if descriptor is None:
        raise ValueError(f'{attr!r} is not a property of self ({self!r})')
    if not isinstance(other, Model):
        raise ValueError("'other' is not a Bokeh model: %r" % other)
    other_descriptor = other.lookup(other_attr, raises=False)
    if other_descriptor is None:
        raise ValueError(f'{other_attr!r} is not a property of other ({other!r})')
    from bokeh.models import CustomJS
    selector = f'[{attr_selector!r}]' if attr_selector is not None else ''
    cb = CustomJS(args=dict(other=other), code=f'other.{other_descriptor.name} = this.{descriptor.name}{selector}')
    self.js_on_change(attr, cb)