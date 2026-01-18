import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg', ['com', 'halflife', 'span', 'alpha'])
def test_ewm_getitem_attributes_retained(arg, adjust, ignore_na):
    kwargs = {arg: 1, 'adjust': adjust, 'ignore_na': ignore_na}
    ewm = DataFrame({'A': range(1), 'B': range(1)}).ewm(**kwargs)
    expected = {attr: getattr(ewm, attr) for attr in ewm._attributes}
    ewm_slice = ewm['A']
    result = {attr: getattr(ewm, attr) for attr in ewm_slice._attributes}
    assert result == expected