import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def data_union_from_json(value, widget):
    """Deserializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, str) and value.startswith('IPY_MODEL_'):
        return widget_serialization['from_json'](value, widget)
    return array_from_json(value, widget)