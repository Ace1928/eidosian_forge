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
def test_arraysequence_copy(self):
    orig = SEQ_DATA['seq']
    seq = orig.copy()
    n_rows = seq.total_nb_rows
    assert n_rows == orig.total_nb_rows
    assert_array_equal(seq._data, orig._data[:n_rows])
    assert seq._data is not orig._data
    assert_array_equal(seq._offsets, orig._offsets)
    assert seq._offsets is not orig._offsets
    assert_array_equal(seq._lengths, orig._lengths)
    assert seq._lengths is not orig._lengths
    assert seq.common_shape == orig.common_shape
    seq = orig[::2].copy()
    check_arr_seq(seq, SEQ_DATA['data'][::2])
    assert seq._data is not orig._data