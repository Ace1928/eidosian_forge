import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_asserts(self):

    def make_cycle():
        a = []
        a.append(a)
        a.append(a)
        return a
    with assert_raises(AssertionError):
        with assert_no_gc_cycles():
            make_cycle()
    with assert_raises(AssertionError):
        assert_no_gc_cycles(make_cycle)