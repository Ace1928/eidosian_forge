import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
@pytest.mark.xfail(reason='Inconsistency between floordiv and divmod; we raise for floordiv but not for divmod. This matches what we do for non-masked bool dtype.')
def test_divmod(self, data):
    super().test_divmod(data)