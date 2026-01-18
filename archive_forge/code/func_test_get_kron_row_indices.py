from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_get_kron_row_indices(self, backend):
    """
        kron(l,r)
        with
        l = [[x1, x3],  r = [[a],
             [x2, x4]]       [b]]

        yields
        [[ax1, ax3],
         [bx1, bx3],
         [ax2, ax4],
         [bx2, bx4]]

        Which is what we get when we compute kron(l,r) directly,
        as l is represented as eye(4) and r is reshaped into a column vector.

        So we have:
        kron(l,r) =
        [[a, 0, 0, 0],
         [b, 0, 0, 0],
         [0, a, 0, 0],
         [0, b, 0, 0],
         [0, 0, a, 0],
         [0, 0, b, 0],
         [0, 0, 0, a],
         [0, 0, 0, b]].

        Thus, this function should return arange(8).
        """
    indices = backend._get_kron_row_indices((2, 2), (2, 1))
    assert np.all(indices == np.arange(8))
    '\n        kron(l,r)\n        with \n        l = [[x1],  r = [[a, c],\n             [x2]]       [b, d]]\n\n        yields\n        [[ax1, cx1],\n         [bx1, dx1],\n         [ax2, cx2],\n         [bx2, dx2]]\n        \n        Here, we have to swap the row indices of the resulting matrix.\n        Immediately applying kron(l,r) gives to eye(2) and r reshaped to \n        a column vector gives.\n                 \n        So we have:\n        kron(l,r) = \n        [[a, 0],\n         [b, 0],\n         [c, 0],\n         [d, 0],\n         [0, a],\n         [0, b]\n         [0, c],\n         [0, d]].\n\n        Thus, we need to to return [0, 1, 4, 5, 2, 3, 6, 7].\n        '
    indices = backend._get_kron_row_indices((2, 1), (2, 2))
    assert np.all(indices == [0, 1, 4, 5, 2, 3, 6, 7])
    indices = backend._get_kron_row_indices((1, 2), (3, 2))
    assert np.all(indices == np.arange(12))
    indices = backend._get_kron_row_indices((3, 2), (1, 2))
    assert np.all(indices == [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11])
    indices = backend._get_kron_row_indices((2, 2), (2, 2))
    expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert np.all(indices == expected)