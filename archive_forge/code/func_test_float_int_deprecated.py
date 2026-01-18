import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('converter', [int, float, complex])
def test_float_int_deprecated(converter):
    with tm.assert_produces_warning(FutureWarning):
        assert converter(Series([1])) == converter(1)