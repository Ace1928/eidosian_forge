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
@pytest.mark.parametrize('data', test_data_categorical_values, ids=test_data_categorical_keys)
@pytest.mark.parametrize('ordered', bool_arg_values, ids=bool_arg_keys)
def test_cat_reorder_categories(data, ordered):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.reorder_categories(list('tades'), ordered=ordered)
    modin_result = modin_series.cat.reorder_categories(list('tades'), ordered=ordered)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)