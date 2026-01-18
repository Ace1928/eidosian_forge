import cirq
import numpy as np
import pytest
def test_matrix_mixture_repr():
    mix = [(0.5, np.array([[1, 0], [0, 1]], dtype=np.dtype('complex64'))), (0.5, np.array([[0, 1], [1, 0]], dtype=np.dtype('complex64')))]
    half_flip = cirq.MixedUnitaryChannel(mix, key='flip')
    assert repr(half_flip) == "cirq.MixedUnitaryChannel(mixture=[(0.5, np.array([[(1+0j), 0j], [0j, (1+0j)]], dtype=np.dtype('complex64'))), (0.5, np.array([[0j, (1+0j)], [(1+0j), 0j]], dtype=np.dtype('complex64')))], key='flip')"