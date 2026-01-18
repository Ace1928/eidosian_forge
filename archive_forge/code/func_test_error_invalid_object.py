from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_error_invalid_object(data, all_arithmetic_operators):
    data, _ = data
    op = all_arithmetic_operators
    opa = getattr(data, op)
    result = opa(pd.DataFrame({'A': data}))
    assert result is NotImplemented
    msg = 'can only perform ops with 1-d structures'
    with pytest.raises(NotImplementedError, match=msg):
        opa(np.arange(len(data)).reshape(-1, len(data)))