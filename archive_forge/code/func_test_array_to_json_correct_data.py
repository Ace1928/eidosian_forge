import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_array_to_json_correct_data():
    data = np.zeros((4, 3), dtype=np.float32)
    json_data = array_to_json(data, None)
    assert json_data == {'buffer': memoryview(data), 'dtype': str(data.dtype), 'shape': (4, 3)}