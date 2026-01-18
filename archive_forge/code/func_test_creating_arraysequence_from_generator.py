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
def test_creating_arraysequence_from_generator(self):
    gen_1, gen_2 = itertools.tee((e for e in SEQ_DATA['data']))
    seq = ArraySequence(gen_1)
    seq_with_buffer = ArraySequence(gen_2, buffer_size=256)
    assert seq_with_buffer.get_data().shape == seq.get_data().shape
    assert seq_with_buffer._buffer_size > seq._buffer_size
    check_arr_seq(seq, SEQ_DATA['data'])
    check_arr_seq(seq_with_buffer, SEQ_DATA['data'])
    check_empty_arr_seq(ArraySequence(gen_1))