import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
def register_mark(key=None):
    """Returns a decorator registering a mark class in the mark type registry.

    If no key is provided, the class name is used as a key. A key is provided
    for each core bqplot mark so that the frontend can use
    this key regardless of the kernel language.
    """

    def wrap(mark):
        name = key if key is not None else mark.__module__ + mark.__name__
        Mark.mark_types[name] = mark
        return mark
    return wrap