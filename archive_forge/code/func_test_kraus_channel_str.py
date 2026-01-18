import cirq
import numpy as np
import pytest
def test_kraus_channel_str():
    ops = [np.array([[1, 1], [1, 1]]) * 0.5, np.array([[1, -1], [-1, 1]]) * 0.5]
    x_meas = cirq.KrausChannel(ops)
    assert str(x_meas) == 'KrausChannel([array([[0.5, 0.5],\n       [0.5, 0.5]]), array([[ 0.5, -0.5],\n       [-0.5,  0.5]])])'
    x_meas_keyed = cirq.KrausChannel(ops, key='x_meas')
    assert str(x_meas_keyed) == 'KrausChannel([array([[0.5, 0.5],\n       [0.5, 0.5]]), array([[ 0.5, -0.5],\n       [-0.5,  0.5]])], key=x_meas)'