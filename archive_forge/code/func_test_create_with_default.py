import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_create_with_default():

    class Foo(HasTraits):
        bar = NDArray(np.zeros((4, 4)))
    foo = Foo()
    np.testing.assert_equal(foo.bar, np.zeros((4, 4)))