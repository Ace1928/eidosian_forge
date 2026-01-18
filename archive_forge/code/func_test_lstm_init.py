import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
def test_lstm_init():
    model = with_padded(LSTM(2, 2, bi=True)).initialize()
    model.initialize()