import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('string', ['Sparse[int, 1]', 'Sparse[float, 0.0]', 'Sparse[bool, True]'])
def test_construct_from_string_fill_value_raises(string):
    with pytest.raises(TypeError, match='fill_value in the string is not'):
        SparseDtype.construct_from_string(string)