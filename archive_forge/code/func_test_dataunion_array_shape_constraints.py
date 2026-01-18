import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_array_shape_constraints():

    class Foo(HasTraits):
        bar = DataUnion(shape_constraint=shape_constraints(None, None, 3))
    raw_data = np.ones((4, 4))
    with pytest.raises(TraitError):
        foo = Foo(bar=raw_data)