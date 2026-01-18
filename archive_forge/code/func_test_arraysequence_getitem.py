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
def test_arraysequence_getitem(self):
    for i, e in enumerate(SEQ_DATA['seq']):
        assert_array_equal(SEQ_DATA['seq'][i], e)
    indices = list(range(len(SEQ_DATA['seq'])))
    seq_view = SEQ_DATA['seq'][indices]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, SEQ_DATA['seq'])
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        seq_view = SEQ_DATA['seq'][np.array(indices, dtype=dtype)]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, SEQ_DATA['seq'])
    SEQ_DATA['rng'].shuffle(indices)
    seq_view = SEQ_DATA['seq'][indices]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, [SEQ_DATA['data'][i] for i in indices])
    seq_view = SEQ_DATA['seq'][::2]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, SEQ_DATA['data'][::2])
    selection = np.array([False, True, True, False, True])
    seq_view = SEQ_DATA['seq'][selection]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, [SEQ_DATA['data'][i] for i, keep in enumerate(selection) if keep])
    with pytest.raises(TypeError):
        SEQ_DATA['seq']['abc']
    seq_view = SEQ_DATA['seq'][:, 2]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, [d[:, 2] for d in SEQ_DATA['data']])
    seq_view = SEQ_DATA['seq'][::-2][:, 2]
    check_arr_seq_view(seq_view, SEQ_DATA['seq'])
    check_arr_seq(seq_view, [d[:, 2] for d in SEQ_DATA['data'][::-2]])