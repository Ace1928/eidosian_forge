import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_custom_var_and_value_name(self, df, value_name, var_name):
    result15 = df.melt(var_name=var_name, value_name=value_name)
    assert result15.columns.tolist() == ['var', 'val']
    result16 = df.melt(id_vars=['id1'], var_name=var_name, value_name=value_name)
    assert result16.columns.tolist() == ['id1', 'var', 'val']
    result17 = df.melt(id_vars=['id1', 'id2'], var_name=var_name, value_name=value_name)
    assert result17.columns.tolist() == ['id1', 'id2', 'var', 'val']
    result18 = df.melt(id_vars=['id1', 'id2'], value_vars='A', var_name=var_name, value_name=value_name)
    assert result18.columns.tolist() == ['id1', 'id2', 'var', 'val']
    result19 = df.melt(id_vars=['id1', 'id2'], value_vars=['A', 'B'], var_name=var_name, value_name=value_name)
    expected19 = DataFrame({'id1': df['id1'].tolist() * 2, 'id2': df['id2'].tolist() * 2, var_name: ['A'] * 10 + ['B'] * 10, value_name: df['A'].tolist() + df['B'].tolist()}, columns=['id1', 'id2', var_name, value_name])
    tm.assert_frame_equal(result19, expected19)
    df20 = df.copy()
    df20.columns.name = 'foo'
    result20 = df20.melt()
    assert result20.columns.tolist() == ['foo', 'value']