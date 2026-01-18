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
@pytest.mark.parametrize('method', ['median', 'skew', 'std', 'sum', 'var', 'prod', 'sem'])
def test_median_skew_std_sum_var_prod_sem_1953(method):
    data = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    arrays = [['1', '1', '1', '2', '2', '2', '3', '3', '3'], ['1', '2', '3', '4', '5', '6', '7', '8', '9']]
    modin_s = pd.Series(data, index=arrays)
    pandas_s = pandas.Series(data, index=arrays)
    eval_general(modin_s, pandas_s, lambda s: getattr(s, method)())