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
def check_arr_seq_view(seq_view, seq):
    assert seq_view._is_view
    assert seq_view is not seq
    assert np.may_share_memory(seq_view._data, seq._data)
    assert seq_view._offsets is not seq._offsets
    assert seq_view._lengths is not seq._lengths