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
def test_creating_arraysequence_from_list(self):
    check_empty_arr_seq(ArraySequence([]))
    N = 5
    for ndim in range(1, N + 1):
        common_shape = tuple([SEQ_DATA['rng'].randint(1, 10) for _ in range(ndim - 1)])
        data = generate_data(nb_arrays=5, common_shape=common_shape, rng=SEQ_DATA['rng'])
        check_arr_seq(ArraySequence(data), data)
    buffer_size = 1.0 / 1024 ** 2
    check_arr_seq(ArraySequence(iter(SEQ_DATA['data']), buffer_size), SEQ_DATA['data'])