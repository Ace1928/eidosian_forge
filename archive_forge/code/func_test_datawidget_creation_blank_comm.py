import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_datawidget_creation_blank_comm(mock_comm):
    try:
        w = NDArrayWidget(comm=mock_comm)
    except TraitError as e:
        assert 'Cannot serialize undefined array' in str(e)
    else:
        assert w.array is Undefined