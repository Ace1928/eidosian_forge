from collections.abc import Iterable, Mapping
from itertools import chain
from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import InstanceDict, TypedTuple
from .widget import register, widget_serialization
from .widget_int import SliderStyle
from .docutils import doc_subst
from traitlets import (Unicode, Bool, Int, Any, Dict, TraitError, CaselessStrEnum,
def findvalue(array, value, compare=lambda x, y: x == y):
    """A function that uses the compare function to return a value from the list."""
    try:
        return next((x for x in array if compare(x, value)))
    except StopIteration:
        raise ValueError('%r not in array' % value)