import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
@pytest.mark.parametrize('ops', [Ops(), NumpyOps()])
@pytest.mark.parametrize('nO,nI', [(1, 2), (2, 2), (100, 200), (9, 6)])
def test_LSTM_init_with_sizes(ops, nO, nI):
    model = with_padded(LSTM(nO, nI, depth=1)).initialize()
    for node in model.walk():
        model.ops = ops
        assert node.has_param('LSTM') is not None
        assert node.has_param('HC0') is not None
    for node in model.walk():
        if node.has_param('LSTM'):
            params = node.get_param('LSTM')
            assert params.shape == (nO * 4 * nI + nO * 4 + (nO * 4 * nO + nO * 4),)
        if node.has_param('HC0'):
            params = node.get_param('HC0')
            assert params.shape == (2, 1, 1, nO)