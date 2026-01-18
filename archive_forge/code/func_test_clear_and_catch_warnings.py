import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_clear_and_catch_warnings():
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    my_mod.__warningregistry__ = {'warning1': 1, 'warning2': 2}
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 2)
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 0)