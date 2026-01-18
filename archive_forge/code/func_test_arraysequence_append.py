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
def test_arraysequence_append(self):
    element = generate_data(nb_arrays=1, common_shape=SEQ_DATA['seq'].common_shape, rng=SEQ_DATA['rng'])[0]
    seq = SEQ_DATA['seq'].copy()
    seq.append(element)
    check_arr_seq(seq, SEQ_DATA['data'] + [element])
    seq = SEQ_DATA['seq'].copy()
    seq.append(element.tolist())
    check_arr_seq(seq, SEQ_DATA['data'] + [element])
    seq = ArraySequence()
    seq.append(element)
    check_arr_seq(seq, [element])
    seq = SEQ_DATA['seq'].copy()
    seq.append([])
    check_arr_seq(seq, SEQ_DATA['seq'])
    element = generate_data(nb_arrays=1, common_shape=SEQ_DATA['seq'].common_shape * 2, rng=SEQ_DATA['rng'])[0]
    with pytest.raises(ValueError):
        seq.append(element)