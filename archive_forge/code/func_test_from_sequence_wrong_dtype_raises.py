import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_from_sequence_wrong_dtype_raises(using_infer_string):
    pytest.importorskip('pyarrow')
    with pd.option_context('string_storage', 'python'):
        ArrowStringArray._from_sequence(['a', None, 'c'], dtype='string')
    with pd.option_context('string_storage', 'pyarrow'):
        ArrowStringArray._from_sequence(['a', None, 'c'], dtype='string')
    with pytest.raises(AssertionError, match=None):
        ArrowStringArray._from_sequence(['a', None, 'c'], dtype='string[python]')
    ArrowStringArray._from_sequence(['a', None, 'c'], dtype='string[pyarrow]')
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context('string_storage', 'python'):
                ArrowStringArray._from_sequence(['a', None, 'c'], dtype=StringDtype())
    with pd.option_context('string_storage', 'pyarrow'):
        ArrowStringArray._from_sequence(['a', None, 'c'], dtype=StringDtype())
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            ArrowStringArray._from_sequence(['a', None, 'c'], dtype=StringDtype('python'))
    ArrowStringArray._from_sequence(['a', None, 'c'], dtype=StringDtype('pyarrow'))
    with pd.option_context('string_storage', 'python'):
        StringArray._from_sequence(['a', None, 'c'], dtype='string')
    with pd.option_context('string_storage', 'pyarrow'):
        StringArray._from_sequence(['a', None, 'c'], dtype='string')
    StringArray._from_sequence(['a', None, 'c'], dtype='string[python]')
    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(['a', None, 'c'], dtype='string[pyarrow]')
    if not using_infer_string:
        with pd.option_context('string_storage', 'python'):
            StringArray._from_sequence(['a', None, 'c'], dtype=StringDtype())
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context('string_storage', 'pyarrow'):
                StringArray._from_sequence(['a', None, 'c'], dtype=StringDtype())
    StringArray._from_sequence(['a', None, 'c'], dtype=StringDtype('python'))
    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(['a', None, 'c'], dtype=StringDtype('pyarrow'))