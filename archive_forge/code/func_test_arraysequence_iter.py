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
def test_arraysequence_iter(self):
    assert_arrays_equal(SEQ_DATA['seq'], SEQ_DATA['data'])
    seq = SEQ_DATA['seq'].copy()
    seq._lengths = seq._lengths[::2]
    with pytest.raises(ValueError):
        list(seq)