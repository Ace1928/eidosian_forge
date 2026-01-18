import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_dtype_coerce():

    class Foo(HasTraits):
        bar = NDArray(dtype=np.uint8)
    foo = Foo(bar=[12.5, 33.2, 1, 4])
    np.testing.assert_equal(foo.bar, np.array([12, 33, 1, 4], dtype=np.uint8))
    with pytest.raises(TraitError):
        foo = Foo(bar=[12.5, 33.2, 1 + 2j, 4])
    foo = Foo(bar=[12.5, -33.2, 1, 4])
    np.testing.assert_equal(foo.bar, np.array([12, 223, 1, 4], dtype=np.uint8))