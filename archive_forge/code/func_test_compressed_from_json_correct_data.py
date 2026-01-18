import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_compressed_from_json_correct_data():
    orig_data = np.zeros((4, 3), dtype=np.float32)
    raw_data = memoryview(zlib.compress(orig_data, 6))
    json_data = {'compressed_buffer': raw_data, 'dtype': 'float32', 'shape': [4, 3]}
    data = array_from_compressed_json(json_data, None)
    np.testing.assert_equal(orig_data, data)