import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
def test_list2padded():
    ops = NumpyOps()
    seqs = [numpy.zeros((5, 4)), numpy.zeros((8, 4)), numpy.zeros((2, 4))]
    padded = ops.list2padded(seqs)
    arr = padded.data
    size_at_t = padded.size_at_t
    assert arr.shape == (8, 3, 4)
    assert size_at_t[0] == 3
    assert size_at_t[1] == 3
    assert size_at_t[2] == 2
    assert size_at_t[3] == 2
    assert size_at_t[4] == 2
    assert size_at_t[5] == 1
    assert size_at_t[6] == 1
    assert size_at_t[7] == 1
    unpadded = ops.padded2list(padded)
    assert unpadded[0].shape == (5, 4)
    assert unpadded[1].shape == (8, 4)
    assert unpadded[2].shape == (2, 4)