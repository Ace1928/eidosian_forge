import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_constrained_datawidget_deprecated():
    with pytest.warns(UserWarning) as warnings:
        ColorImage = ConstrainedNDArrayWidget(shape_constraints(None, None, 3), dtype=np.uint8)
        assert len(warnings) > 0
        for warn in warnings:
            assert 'ConstrainedNDArrayWidget is deprecated' in str(warn.message)
    with pytest.warns(UserWarning) as warnings:
        with pytest.raises(TraitError):
            ColorImage(np.zeros((4, 4)))
        w = ColorImage(np.zeros((4, 4, 3)))
        for warn in warnings:
            assert 'Given trait value dtype "float64" does not match required type "uint8"' in str(warn.message)
    np.testing.assert_equal(w.array, np.zeros((4, 4, 3), dtype=np.uint8))