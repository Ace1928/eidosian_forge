from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_float_like(floating):
    assert isinstance(float_like(floating, 'floating'), float)
    assert isinstance(float_like(floating, 'floating', optional=True), float)
    assert float_like(None, 'floating', optional=True) is None
    if isinstance(floating, (int, np.integer, float, np.inexact)):
        assert isinstance(float_like(floating, 'floating', strict=True), float)
        assert float_like(None, 'floating', optional=True, strict=True) is None