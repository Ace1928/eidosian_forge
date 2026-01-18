from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.fixture(params=['mixed', 'empty'])
def biggie_df_fixture(request):
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == 'mixed':
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(200), 'B': Index([f'{i}?!' for i in range(200)])}, index=np.arange(200))
        df.loc[:20, 'A'] = np.nan
        df.loc[:20, 'B'] = np.nan
        return df
    elif request.param == 'empty':
        df = DataFrame(index=np.arange(200))
        return df