import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def array_from_compressed_json(value, widget):
    """Compressed array JSON de-serializer."""
    comp = value.pop('compressed_buffer', None) if value is not None else None
    if comp is not None:
        value['buffer'] = zlib.decompress(comp)
    return array_from_json(value, widget)