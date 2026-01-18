import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_hold_sync_segment(mock_comm):
    data = np.zeros((2, 4))
    w = NDArrayWidget(data)
    w.comm = mock_comm
    with w.hold_sync():
        data.ravel()[:4] = 1
        w.sync_segment([(0, 4)])
        assert len(mock_comm.log_send) == 0
    assert 1 <= len(mock_comm.log_send) <= 2
    buffers = mock_comm.log_send[0][1]['buffers']
    assert len(buffers) == 1
    np.testing.assert_equal(buffers[0], memoryview(data.ravel()[:4]))