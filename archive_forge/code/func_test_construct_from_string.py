import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('string, expected', [('Sparse[float64]', SparseDtype(np.dtype('float64'))), ('Sparse[float32]', SparseDtype(np.dtype('float32'))), ('Sparse[int]', SparseDtype(np.dtype('int'))), ('Sparse[str]', SparseDtype(np.dtype('str'))), ('Sparse[datetime64[ns]]', SparseDtype(np.dtype('datetime64[ns]'))), ('Sparse', SparseDtype(np.dtype('float'), np.nan))])
def test_construct_from_string(string, expected):
    result = SparseDtype.construct_from_string(string)
    assert result == expected