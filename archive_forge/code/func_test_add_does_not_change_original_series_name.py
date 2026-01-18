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
def test_add_does_not_change_original_series_name():
    s1 = pd.Series(1, name=1)
    s2 = pd.Series(2, name=2)
    original_s1 = s1.copy(deep=True)
    original_s2 = s2.copy(deep=True)
    _ = s1 + s2
    df_equals(s1, original_s1)
    df_equals(s2, original_s2)