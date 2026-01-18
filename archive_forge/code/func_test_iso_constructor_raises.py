from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('fmt', ['PPPPPPPPPPPP', 'PDTHMS', 'P0DT999H999M999S', 'P1DT0H0M0.0000000000000S', 'P1DT0H0M0.S', 'P', '-P'])
def test_iso_constructor_raises(fmt):
    msg = f'Invalid ISO 8601 Duration format - {fmt}'
    with pytest.raises(ValueError, match=msg):
        Timedelta(fmt)