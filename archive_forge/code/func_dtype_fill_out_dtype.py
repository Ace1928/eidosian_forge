from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
@pytest.fixture(params=[(np.int8, np.int16(127), np.int8), (np.int8, np.int16(128), np.int16), (np.int32, 1, np.int32), (np.int32, 2.0, np.float64), (np.int32, 3.0 + 4j, np.complex128), (np.int32, True, np.object_), (np.int32, '', np.object_), (np.float64, 1, np.float64), (np.float64, 2.0, np.float64), (np.float64, 3.0 + 4j, np.complex128), (np.float64, True, np.object_), (np.float64, '', np.object_), (np.complex128, 1, np.complex128), (np.complex128, 2.0, np.complex128), (np.complex128, 3.0 + 4j, np.complex128), (np.complex128, True, np.object_), (np.complex128, '', np.object_), (np.bool_, 1, np.object_), (np.bool_, 2.0, np.object_), (np.bool_, 3.0 + 4j, np.object_), (np.bool_, True, np.bool_), (np.bool_, '', np.object_)])
def dtype_fill_out_dtype(request):
    return request.param