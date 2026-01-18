from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_int_like(integer):
    assert isinstance(int_like(integer, 'integer'), int)
    assert isinstance(int_like(integer, 'integer', optional=True), int)
    assert int_like(None, 'floating', optional=True) is None
    if isinstance(integer, (int, np.integer)):
        assert isinstance(int_like(integer, 'integer', strict=True), int)
        assert int_like(None, 'floating', optional=True, strict=True) is None