import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_downcast():
    frame_data = {'a': [1.0, np.nan]}
    df = pandas.DataFrame(frame_data)
    result = df.fillna(0, downcast='infer')
    modin_df = pd.DataFrame(frame_data).fillna(0, downcast='infer')
    df_equals(modin_df, result)
    df = pandas.DataFrame(frame_data)
    result = df.fillna({'a': 0}, downcast='infer')
    modin_df = pd.DataFrame(frame_data).fillna({'a': 0}, downcast='infer')
    df_equals(modin_df, result)