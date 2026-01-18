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
def dt_with_empty_partition(lib):
    df = pd.concat([pd.DataFrame([None]), pd.DataFrame([pd.to_timedelta(1)])], axis=1).dropna(axis=1).squeeze(1)
    if isinstance(df, pd.DataFrame) and get_current_execution() != 'BaseOnPython' and (StorageFormat.get() != 'Hdk'):
        assert df._query_compiler._modin_frame._partitions.shape == (1, 2)
    return df.dt.days