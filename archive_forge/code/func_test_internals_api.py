import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import internals
from pandas.core.internals import api
def test_internals_api():
    assert internals.make_block is api.make_block