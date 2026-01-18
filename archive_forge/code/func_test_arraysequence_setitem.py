import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def test_arraysequence_setitem(self):
    seq = SEQ_DATA['seq'] * 0
    for i, e in enumerate(SEQ_DATA['seq']):
        seq[i] = e
    check_arr_seq(seq, SEQ_DATA['seq'])
    seq = SEQ_DATA['seq'].copy()
    seq[:] = 0
    assert seq._data.sum() == 0
    seq = SEQ_DATA['seq'] * 0
    seq[:] = SEQ_DATA['data']
    check_arr_seq(seq, SEQ_DATA['data'])
    seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
    seq[:, 0] = 0
    assert seq._data[:, 0].sum() == 0
    seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
    seq[range(len(seq))] = 0
    assert seq._data.sum() == 0
    seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
    seq[0:4] = seq[5:9]
    check_arr_seq(seq[0:4], seq[5:9])
    seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
    with pytest.raises(ValueError):
        seq[0:4] = seq[5:10]
    seq1 = ArraySequence(np.arange(10).reshape(5, 2))
    seq2 = ArraySequence(np.arange(15).reshape(5, 3))
    with pytest.raises(ValueError):
        seq1[0:5] = seq2
    seq1 = ArraySequence(np.arange(12).reshape(2, 2, 3))
    seq2 = ArraySequence(np.arange(8).reshape(2, 2, 2))
    with pytest.raises(ValueError):
        seq1[0:2] = seq2
    with pytest.raises(TypeError):
        seq[object()] = None