import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_array_to_json_continuous_data():
    data = np.zeros((4, 3), dtype=np.float32, order='F')
    json_data = array_to_json(data, None)
    reinterpreted_data = array_from_json(json_data, None)
    np.testing.assert_equal(data, reinterpreted_data)
    assert reinterpreted_data.flags['C_CONTIGUOUS']