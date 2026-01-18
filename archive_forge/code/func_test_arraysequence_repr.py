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
def test_arraysequence_repr(self):
    repr(SEQ_DATA['seq'])
    nb_arrays = 50
    seq = ArraySequence(generate_data(nb_arrays, common_shape=(1,), rng=SEQ_DATA['rng']))
    bkp_threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=nb_arrays * 2)
    txt1 = repr(seq)
    np.set_printoptions(threshold=nb_arrays // 2)
    txt2 = repr(seq)
    assert len(txt2) < len(txt1)
    np.set_printoptions(threshold=bkp_threshold)