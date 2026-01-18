import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_get_union_array_with_widget():

    class Foo(Widget):
        bar = DataUnion()
    raw_data = np.ones((4, 4))
    foo = Foo(bar=NDArrayWidget(raw_data))
    assert get_union_array(foo.bar) is raw_data