import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_new_keyword_only(self):

    @deprecate_parameter('old', new_name='new', start_version='0.19', stop_version='0.21')
    def foo(arg0, old=DEPRECATED, *, new=1, arg3=None):
        """Expected docstring"""
        return (arg0, new, arg3)
    with warnings.catch_warnings(record=True) as recorded:
        assert foo(0) == (0, 1, None)
        assert foo(0, new=1, arg3=2) == (0, 1, 2)
        assert foo(0, new=2) == (0, 2, None)
        assert foo(0, arg3=2) == (0, 1, 2)
    assert len(recorded) == 0