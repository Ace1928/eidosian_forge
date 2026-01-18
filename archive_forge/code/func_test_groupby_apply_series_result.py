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
@pytest.mark.parametrize('modify_config', [{RangePartitioning: True}], indirect=True)
def test_groupby_apply_series_result(modify_config):
    df = pd.DataFrame(np.random.randint(5, 10, size=5), index=[f's{i + 1}' for i in range(5)])
    df['group'] = [1, 1, 2, 2, 3]
    eval_general(df, df._to_pandas(), lambda df: df.groupby('group').apply(lambda x: x.name + 2))