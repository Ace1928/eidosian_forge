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
def test_arraysequence_extend(self):
    new_data = generate_data(nb_arrays=10, common_shape=SEQ_DATA['seq'].common_shape, rng=SEQ_DATA['rng'])
    seq = SEQ_DATA['seq'].copy()
    seq.extend([])
    check_arr_seq(seq, SEQ_DATA['data'])
    seq = SEQ_DATA['seq'].copy()
    seq.extend(new_data)
    check_arr_seq(seq, SEQ_DATA['data'] + new_data)
    seq = SEQ_DATA['seq'].copy()
    seq.extend((d for d in new_data))
    check_arr_seq(seq, SEQ_DATA['data'] + new_data)
    seq = SEQ_DATA['seq'].copy()
    seq.extend(ArraySequence(new_data))
    check_arr_seq(seq, SEQ_DATA['data'] + new_data)
    seq = SEQ_DATA['seq'].copy()
    seq.extend(ArraySequence(new_data)[::2])
    check_arr_seq(seq, SEQ_DATA['data'] + new_data[::2])
    seq = ArraySequence()
    seq.extend(ArraySequence())
    check_empty_arr_seq(seq)
    seq.extend(SEQ_DATA['seq'])
    check_arr_seq(seq, SEQ_DATA['data'])
    data = generate_data(nb_arrays=10, common_shape=SEQ_DATA['seq'].common_shape * 2, rng=SEQ_DATA['rng'])
    seq = SEQ_DATA['seq'].copy()
    with pytest.raises(ValueError):
        seq.extend(data)
    working_slice = seq[:2]
    seq.extend(ArraySequence(new_data))