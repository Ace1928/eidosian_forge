import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_compressed_to_compressed_json():
    data = np.zeros((4, 3), dtype=np.float32)
    dummy = Widget()
    dummy.compression_level = 6
    json_data = array_to_compressed_json(data, dummy)
    assert tuple(sorted(json_data.keys())) == ('compressed_buffer', 'dtype', 'shape')
    assert json_data['shape'] == (4, 3)
    assert json_data['dtype'] == str(data.dtype)
    comp = json_data['compressed_buffer']
    zlib.decompress(comp)