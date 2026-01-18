import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['aggregate', 'std', 'corr', 'cov', 'var'])
def test_ewm_notimplementederror_raises(self, method):
    ser = Series(range(10))
    kwargs = {}
    if method == 'aggregate':
        kwargs['func'] = lambda x: x
    with pytest.raises(NotImplementedError, match='.* is not implemented.'):
        getattr(ser.ewm(1).online(), method)(**kwargs)