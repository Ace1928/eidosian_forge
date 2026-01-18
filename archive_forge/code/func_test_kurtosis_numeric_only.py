from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('numeric_only', [False, True])
def test_kurtosis_numeric_only(axis, numeric_only):
    expected_exception = None
    if axis:
        expected_exception = ValueError('No axis named columns for object type Series')
    eval_general(*create_test_series(test_data_diff_dtype), lambda df: df.kurtosis(axis=axis, numeric_only=numeric_only), expected_exception=expected_exception)