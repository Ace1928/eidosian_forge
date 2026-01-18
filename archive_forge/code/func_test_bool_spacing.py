import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_bool_spacing(self):
    assert_equal(repr(np.array([True, True])), 'array([ True,  True])')
    assert_equal(repr(np.array([True, False])), 'array([ True, False])')
    assert_equal(repr(np.array([True])), 'array([ True])')
    assert_equal(repr(np.array(True)), 'array(True)')
    assert_equal(repr(np.array(False)), 'array(False)')