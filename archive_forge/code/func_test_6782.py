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
def test_6782():
    datetime_scalar = datetime.datetime(1970, 1, 1, 0, 0)
    with pytest.warns(UserWarning) as warns:
        _ = pd.Series([datetime.datetime(2000, 1, 1)]) - datetime_scalar
        for warn in warns.list:
            assert 'Adding/subtracting object-dtype array to DatetimeArray not vectorized' not in str(warn)