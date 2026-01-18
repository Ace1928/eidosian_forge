import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['Y', 'M', 'ps', 'fs', 'as'])
def test_ints_to_pytimedelta_unsupported(unit):
    arr = np.arange(6, dtype=np.int64).view(f'm8[{unit}]')
    with pytest.raises(NotImplementedError, match='\\d{1,2}'):
        ints_to_pytimedelta(arr, box=False)
    msg = "Only resolutions 's', 'ms', 'us', 'ns' are supported"
    with pytest.raises(NotImplementedError, match=msg):
        ints_to_pytimedelta(arr, box=True)