import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_finalize_frame_series_name():
    df = pd.DataFrame({'name': [1, 2]})
    result = pd.Series([1, 2]).__finalize__(df)
    assert result.name is None