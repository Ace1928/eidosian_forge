import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def data_union_to_json(value, widget):
    """Serializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, Widget):
        return widget_serialization['to_json'](value, widget)
    return array_to_json(value, widget)