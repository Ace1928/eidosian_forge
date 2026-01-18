from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def get_series_na():
    return [Series([np.nan], dtype='Int64'), Series([np.nan], dtype='float'), Series([np.nan], dtype='object'), Series([pd.NaT])]