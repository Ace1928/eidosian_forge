import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
@pytest.mark.skipif(not has_cupy_gpu, reason='needs CuPy GPU')
def test_model_gpu():
    pytest.importorskip('ml_datasets')
    import ml_datasets
    with use_ops('cupy'):
        n_hidden = 32
        dropout = 0.2
        (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
        model = chain(Relu(nO=n_hidden, dropout=dropout), Relu(nO=n_hidden, dropout=dropout), Softmax())
        train_X = model.ops.asarray(train_X)
        train_Y = model.ops.asarray(train_Y)
        dev_X = model.ops.asarray(dev_X)
        dev_Y = model.ops.asarray(dev_Y)
        model.initialize(X=train_X[:5], Y=train_Y[:5])
        optimizer = Adam(0.001)
        batch_size = 128
        for i in range(2):
            batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
            for X, Y in batches:
                Yh, backprop = model.begin_update(X)
                backprop(Yh - Y)
                model.finish_update(optimizer)
            correct = 0
            total = 0
            for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
                Yh = model.predict(X)
                correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
                total += Yh.shape[0]