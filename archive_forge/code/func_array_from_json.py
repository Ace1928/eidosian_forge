import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def array_from_json(value, widget):
    """Array JSON de-serializer."""
    if value is None:
        return None
    n = np.frombuffer(value['buffer'], dtype=value['dtype'])
    n.shape = value['shape']
    return n