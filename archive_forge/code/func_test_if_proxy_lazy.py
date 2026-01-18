import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_if_proxy_lazy(self):
    """Verify that proxy is able to pass simple comparison checks without triggering materialization."""
    lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()
    assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
    assert not lazy_proxy._is_materialized
    assert lazy_proxy == 'category'
    assert isinstance(lazy_proxy, pd.CategoricalDtype)
    assert isinstance(lazy_proxy, pandas.CategoricalDtype)
    assert str(lazy_proxy) == 'category'
    assert str(lazy_proxy) == str(actual_dtype)
    assert not lazy_proxy.ordered
    assert not lazy_proxy._is_materialized
    assert lazy_proxy == actual_dtype
    assert actual_dtype == lazy_proxy
    assert repr(lazy_proxy) == repr(actual_dtype)
    assert lazy_proxy.categories.equals(actual_dtype.categories)
    assert lazy_proxy._is_materialized