from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.fixture
def disable_bottleneck(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(nanops, '_USE_BOTTLENECK', False)
        yield