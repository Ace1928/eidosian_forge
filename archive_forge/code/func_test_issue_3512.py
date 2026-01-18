import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
def test_issue_3512():
    data = np.random.rand(129)
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_ans = modin_df[0:33].rolling(window=21).mean()
    pandas_ans = pandas_df[0:33].rolling(window=21).mean()
    df_equals(modin_ans, pandas_ans)