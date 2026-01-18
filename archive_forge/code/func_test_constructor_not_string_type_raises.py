import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('chunked', [True, False])
@pytest.mark.parametrize('array', ['numpy', 'pyarrow'])
def test_constructor_not_string_type_raises(array, chunked, arrow_string_storage):
    pa = pytest.importorskip('pyarrow')
    array = pa if array in arrow_string_storage else np
    arr = array.array([1, 2, 3])
    if chunked:
        if array is np:
            pytest.skip('chunked not applicable to numpy array')
        arr = pa.chunked_array(arr)
    if array is np:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"
    else:
        msg = re.escape('ArrowStringArray requires a PyArrow (chunked) array of large_string type')
    with pytest.raises(ValueError, match=msg):
        ArrowStringArray(arr)