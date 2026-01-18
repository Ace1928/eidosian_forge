import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def array_to_compressed_json(value, widget):
    """Compressed array JSON serializer."""
    state = array_to_json(value, widget)
    if state is None:
        return state
    compression = getattr(widget, 'compression_level', 0)
    if compression == 0:
        return state
    buffer = state.pop('buffer')
    state['compressed_buffer'] = zlib.compress(buffer, compression)
    return state