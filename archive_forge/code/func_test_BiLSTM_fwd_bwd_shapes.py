import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
@pytest.mark.parametrize('ops', [Ops(), NumpyOps()])
@pytest.mark.parametrize('nO,nI,depth,bi,lengths', [(1, 1, 1, False, [1]), (12, 32, 1, False, [3, 1]), (2, 2, 1, True, [2, 5, 7]), (2, 2, 2, False, [7, 2, 4]), (2, 2, 2, True, [1]), (32, 16, 1, True, [5, 1, 10, 2]), (32, 16, 2, True, [3, 3, 5]), (32, 16, 3, True, [9, 2, 4])])
def test_BiLSTM_fwd_bwd_shapes(ops, nO, nI, depth, bi, lengths):
    Xs = [numpy.ones((length, nI), dtype='f') for length in lengths]
    model = with_padded(LSTM(nO, nI, depth=depth, bi=bi)).initialize(X=Xs)
    for node in model.walk():
        node.ops = ops
    ys, backprop_ys = model(Xs, is_train=True)
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack(Xs).shape