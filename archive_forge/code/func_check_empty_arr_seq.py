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
def check_empty_arr_seq(seq):
    assert len(seq) == 0
    assert len(seq._offsets) == 0
    assert len(seq._lengths) == 0
    assert seq._data.ndim == 1
    assert seq.common_shape == ()