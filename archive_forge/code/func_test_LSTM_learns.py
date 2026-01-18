import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
def test_LSTM_learns():
    fix_random_seed(0)
    nO = 2
    nI = 2

    def sgd(key, weights, gradient):
        weights -= 0.001 * gradient
        return (weights, gradient * 0)
    model = with_padded(LSTM(nO, nI))
    X = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    Y = [[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]
    X = [model.ops.asarray(x, dtype='f').reshape((1, -1)) for x in X]
    Y = [model.ops.asarray(y, dtype='f').reshape((1, -1)) for y in Y]
    model = model.initialize(X, Y)
    Yhs, bp_Yhs = model.begin_update(X)
    loss1 = sum([((yh - y) ** 2).sum() for yh, y in zip(Yhs, Y)])
    Yhs, bp_Yhs = model.begin_update(X)
    dYhs = [yh - y for yh, y in zip(Yhs, Y)]
    dXs = bp_Yhs(dYhs)
    model.finish_update(sgd)
    Yhs, bp_Yhs = model.begin_update(X)
    dYhs = [yh - y for yh, y in zip(Yhs, Y)]
    dXs = bp_Yhs(dYhs)
    loss2 = sum([((yh - y) ** 2).sum() for yh, y in zip(Yhs, Y)])
    assert loss1 > loss2, (loss1, loss2)