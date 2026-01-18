from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names_in,names_out', [(None, [('L1', 'x'), ('L2', 'y')]), (['x', 'y'], ['x', 'y'])])
def test_from_frame_valid_names(names_in, names_out):
    df = pd.DataFrame([['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b']], columns=MultiIndex.from_tuples([('L1', 'x'), ('L2', 'y')]))
    mi = MultiIndex.from_frame(df, names=names_in)
    assert mi.names == names_out