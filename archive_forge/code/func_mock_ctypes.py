from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.fixture
def mock_ctypes(monkeypatch):
    """
    Mocks WinError to help with testing the clipboard.
    """

    def _mock_win_error():
        return 'Window Error'
    with monkeypatch.context() as m:
        m.setattr('ctypes.WinError', _mock_win_error, raising=False)
        yield