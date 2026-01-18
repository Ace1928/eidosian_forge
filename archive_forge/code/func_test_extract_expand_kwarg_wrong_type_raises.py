from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_kwarg_wrong_type_raises(any_string_dtype):
    values = Series(['fooBAD__barBAD', np.nan, 'foo'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='expand must be True or False'):
        values.str.extract('.*(BAD[_]+).*(BAD)', expand=None)