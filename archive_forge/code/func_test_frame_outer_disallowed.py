from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
def test_frame_outer_disallowed():
    df = pd.DataFrame({'A': [1, 2]})
    with pytest.raises(NotImplementedError, match=''):
        np.subtract.outer(df, df)