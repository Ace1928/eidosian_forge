import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_unicode(self, sparse):
    e = 'e'
    eacute = unicodedata.lookup('LATIN SMALL LETTER E WITH ACUTE')
    s = [e, eacute, eacute]
    res = get_dummies(s, prefix='letter', sparse=sparse)
    exp = DataFrame({'letter_e': [True, False, False], f'letter_{eacute}': [False, True, True]})
    if sparse:
        exp = exp.apply(SparseArray, fill_value=False)
    tm.assert_frame_equal(res, exp)