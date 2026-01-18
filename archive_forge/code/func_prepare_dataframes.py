import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def prepare_dataframes(df):
    df = (pd if isinstance(df, pd.DataFrame) else pandas).concat([df, df], axis=0)
    df = df.reset_index(drop=True)
    df = df.join(df, rsuffix='_y')
    return df.set_index(['class', 'animal', 'locomotion'])