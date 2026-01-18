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
def set_pyqt_clipboard(monkeypatch):
    qt_cut, qt_paste = init_qt_clipboard()
    with monkeypatch.context() as m:
        m.setattr(pd.io.clipboard, 'clipboard_set', qt_cut)
        m.setattr(pd.io.clipboard, 'clipboard_get', qt_paste)
        yield