from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_index_one_two_groups():
    s = Series(['a3', 'b3', 'd4c2'], index=['A3', 'B3', 'D4'], name='series_name')
    r = s.index.str.extract('([A-Z])', expand=True)
    e = DataFrame(['A', 'B', 'D'])
    tm.assert_frame_equal(r, e)
    r = s.index.str.extract('(?P<letter>[A-Z])(?P<digit>[0-9])', expand=True)
    e_list = [('A', '3'), ('B', '3'), ('D', '4')]
    e = DataFrame(e_list, columns=['letter', 'digit'])
    tm.assert_frame_equal(r, e)