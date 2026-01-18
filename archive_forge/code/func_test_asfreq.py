import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_asfreq():
    index = pd.date_range('1/1/2000', periods=4, freq='min')
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    df = pd.DataFrame({'s': series})
    with warns_that_defaulting_to_pandas():
        df.asfreq(freq='30S')