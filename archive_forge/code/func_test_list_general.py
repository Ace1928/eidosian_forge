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
def test_list_general():
    pa = pytest.importorskip('pyarrow')
    modin_series, pandas_series = create_test_series([[1, 2, 3], [3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    eval_general(modin_series, pandas_series, lambda series: series.list.flatten())
    eval_general(modin_series, pandas_series, lambda series: series.list.len())
    eval_general(modin_series, pandas_series, lambda series: series.list[0])