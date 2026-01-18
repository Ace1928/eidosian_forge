import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('op', [*('add', 'radd', 'sub', 'rsub', 'mod', 'rmod', 'pow', 'rpow'), *('truediv', 'rtruediv', 'mul', 'rmul', 'floordiv', 'rfloordiv')])
def test_math_functions_level(op):
    modin_df = pd.DataFrame(test_data['int_data'])
    modin_df.index = pandas.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in modin_df.index])
    with warns_that_defaulting_to_pandas():
        getattr(modin_df, op)(modin_df, axis=0, level=1)