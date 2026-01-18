import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_constricts_widget_data():

    class Foo(HasTraits):
        bar = DataUnion(shape_constraint=shape_constraints(None, None, 3))
    ok_data = np.ones((4, 2, 3))
    bad_data = np.ones((4, 4))
    w = NDArrayWidget(ok_data)
    foo = Foo(bar=w)
    with pytest.raises(TraitError):
        w.array = bad_data
    foo.bar = ok_data
    w.array = bad_data