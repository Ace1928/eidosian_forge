from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_attrdict():
    d = _idl.AttrDict({'one': 1})
    assert d['one'] == 1
    assert d.one == 1
    with pytest.raises(KeyError):
        d['two']
    with pytest.raises(AttributeError, match='has no attribute'):
        d.two