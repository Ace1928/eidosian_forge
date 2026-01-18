import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_mixed_use():

    class Foo(HasTraits):
        bar = DataUnion()
    raw_data = np.array([range(5), range(5, 10)], dtype=np.float32)
    data_copy = np.copy(raw_data)
    foo = Foo(bar=raw_data)
    assert foo.bar is raw_data
    w = NDArrayWidget(raw_data)
    foo.bar = w
    assert foo.bar is w
    assert foo.bar.array is raw_data
    np.testing.assert_equal(raw_data, data_copy)