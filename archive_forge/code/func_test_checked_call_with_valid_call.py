from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.usefixtures('mock_ctypes')
def test_checked_call_with_valid_call(monkeypatch):
    """
    Give CheckCall a function that returns a truthy value and
    mock get_errno so it returns true so an exception is not raised.
    The function should return the results from _return_true.
    """

    def _return_true():
        return True
    monkeypatch.setattr('pandas.io.clipboard.get_errno', lambda: False)
    checked_call = CheckedCall(_return_true)
    assert checked_call() is True