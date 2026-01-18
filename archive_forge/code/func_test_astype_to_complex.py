from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
@pytest.mark.parametrize('complex_dtype', [np.complex64, np.complex128])
def test_astype_to_complex(self, complex_dtype, simple_index):
    result = simple_index.astype(complex_dtype)
    assert type(result) is Index and result.dtype == complex_dtype