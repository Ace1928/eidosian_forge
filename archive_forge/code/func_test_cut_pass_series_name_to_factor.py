import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_pass_series_name_to_factor():
    name = 'foo'
    ser = Series(np.random.default_rng(2).standard_normal(100), name=name)
    factor = cut(ser, 4)
    assert factor.name == name