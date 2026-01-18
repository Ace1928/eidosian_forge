import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def lazy_frame(self):
    donor_obj = pd.DataFrame()._query_compiler
    self._mock_obj = mock.patch(f'{donor_obj.__module__}.{donor_obj.__class__.__name__}.lazy_execution', new_callable=mock.PropertyMock)
    patch_obj = self._mock_obj.__enter__()
    patch_obj.return_value = True
    df = pd.DataFrame(**self._df_kwargs)
    assert df._query_compiler.lazy_execution
    return df