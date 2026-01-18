import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_disallowed_none():

    class Foo(HasTraits):
        bar = NDArray(default_value=None)
    foo = Foo(bar=[1, 2, 3])
    assert foo.bar is not None
    with pytest.raises(TraitError):
        foo = Foo(bar=None)
    with pytest.raises(TraitError):
        foo = Foo()
        assert foo.bar is None