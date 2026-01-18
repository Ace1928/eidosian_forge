import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def setup_cache():
    if has_index_cache:
        modin_df1.index
        modin_df2.index
        assert modin_df1._query_compiler._modin_frame.has_index_cache
        assert modin_df2._query_compiler._modin_frame.has_index_cache
    else:
        modin_df1.index = modin_df1.index
        modin_df1._to_pandas()
        modin_df1._query_compiler._modin_frame.set_index_cache(None)
        modin_df2.index = modin_df2.index
        modin_df2._to_pandas()
        modin_df2._query_compiler._modin_frame.set_index_cache(None)