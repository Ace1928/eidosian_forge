import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_datawidget_creation():
    data = np.zeros((2, 4))
    w = NDArrayWidget(data)
    assert w.array is data