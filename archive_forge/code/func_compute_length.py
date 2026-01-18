import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
def compute_length(indices, length):
    if not isinstance(indices, slice):
        return len(indices)
    return compute_sliced_len(indices, length)