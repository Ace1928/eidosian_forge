import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
def test_from_sparse_dtype():
    dtype = SparseDtype('float', 0)
    result = SparseDtype(dtype)
    assert result.fill_value == 0