import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_map_list():
    nI = 4
    nO = 9
    Xs = [numpy.zeros((6, nI), dtype='f'), numpy.ones((3, nI), dtype='f')]
    Y_shapes = [(x.shape[0], nO) for x in Xs]
    model = map_list(Linear())
    model.initialize(X=Xs, Y=[numpy.zeros(shape, dtype='f') for shape in Y_shapes])
    Ys, backprop = model(Xs, is_train=True)
    assert isinstance(Ys, list)
    assert len(Ys) == len(Xs)
    layer = model.layers[0]
    for X, Y in zip(Xs, Ys):
        assert_allclose(layer.predict(X), Y)
    dXs = backprop(Ys)
    assert isinstance(dXs, list)
    assert len(dXs) == len(Xs)
    assert dXs[0].shape == Xs[0].shape
    assert dXs[1].shape == Xs[1].shape