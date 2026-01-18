import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_category_mismatch(check_categorical, using_infer_string):
    if using_infer_string:
        dtype = 'string'
    else:
        dtype = 'object'
    msg = f"""Index are different\n\nAttribute "dtype" are different\n\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, categories_dtype={dtype}\\)\n\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], ordered=False, categories_dtype={dtype}\\)"""
    idx1 = Index(Categorical(['a', 'b']))
    idx2 = Index(Categorical(['a', 'b'], categories=['a', 'b', 'c']))
    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)