import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
def test_extract_ordinals_raises(self):
    arr = np.arange(5)
    freq = to_offset('D')
    with pytest.raises(TypeError, match='values must be object-dtype'):
        extract_ordinals(arr, freq)