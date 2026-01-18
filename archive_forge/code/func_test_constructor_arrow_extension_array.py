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
def test_constructor_arrow_extension_array():
    pa = pytest.importorskip('pyarrow')
    array = pd.arrays.ArrowExtensionArray(pa.array([{'1': '2'}, {'10': '20'}, None], type=pa.map_(pa.string(), pa.string())))
    md_ser, pd_ser = create_test_series(array)
    df_equals(md_ser, pd_ser)
    df_equals(md_ser.dtypes, pd_ser.dtypes)