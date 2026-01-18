import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_clear_and_catch_warnings_inherit():
    my_mod = _get_fresh_mod()
    with my_cacw():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})