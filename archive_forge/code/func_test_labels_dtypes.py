import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_labels_dtypes():
    i = MultiIndex.from_tuples([('A', 1), ('A', 2)])
    assert i.codes[0].dtype == 'int8'
    assert i.codes[1].dtype == 'int8'
    i = MultiIndex.from_product([['a'], range(40)])
    assert i.codes[1].dtype == 'int8'
    i = MultiIndex.from_product([['a'], range(400)])
    assert i.codes[1].dtype == 'int16'
    i = MultiIndex.from_product([['a'], range(40000)])
    assert i.codes[1].dtype == 'int32'
    i = MultiIndex.from_product([['a'], range(1000)])
    assert (i.codes[0] >= 0).all()
    assert (i.codes[1] >= 0).all()