import cirq
import numpy as np
import pytest
def test_matrix_mixture_str():
    mix = [(0.5, np.array([[1, 0], [0, 1]])), (0.5, np.array([[0, 1], [1, 0]]))]
    half_flip = cirq.MixedUnitaryChannel(mix)
    assert str(half_flip) == 'MixedUnitaryChannel([(0.5, array([[1, 0],\n       [0, 1]])), (0.5, array([[0, 1],\n       [1, 0]]))])'
    half_flip_keyed = cirq.MixedUnitaryChannel(mix, key='flip')
    assert str(half_flip_keyed) == 'MixedUnitaryChannel([(0.5, array([[1, 0],\n       [0, 1]])), (0.5, array([[0, 1],\n       [1, 0]]))], key=flip)'