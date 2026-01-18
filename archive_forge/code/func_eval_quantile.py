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
def eval_quantile(modin_groupby, pandas_groupby):
    try:
        pandas_result = pandas_groupby.quantile(q=0.4, numeric_only=True)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_groupby.quantile(q=0.4, numeric_only=True)
    else:
        df_equals(modin_groupby.quantile(q=0.4, numeric_only=True), pandas_result)