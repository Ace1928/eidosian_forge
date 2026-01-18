import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_non_subscriptable_keys():
    frame_data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 6))
    df = pd.DataFrame(frame_data).add_prefix('col')
    pdf = pandas.DataFrame(frame_data).add_prefix('col')
    modin_dict = {'c': df.copy(), 'b': df.copy()}
    pandas_dict = {'c': pdf.copy(), 'b': pdf.copy()}
    modin_result = pd.concat(modin_dict.values(), keys=modin_dict.keys())
    pandas_result = pandas.concat(pandas_dict.values(), keys=pandas_dict.keys())
    df_equals(modin_result, pandas_result)