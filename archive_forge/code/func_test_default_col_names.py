import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_default_col_names(self, df):
    result = df.melt()
    assert result.columns.tolist() == ['variable', 'value']
    result1 = df.melt(id_vars=['id1'])
    assert result1.columns.tolist() == ['id1', 'variable', 'value']
    result2 = df.melt(id_vars=['id1', 'id2'])
    assert result2.columns.tolist() == ['id1', 'id2', 'variable', 'value']