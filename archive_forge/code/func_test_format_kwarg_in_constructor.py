import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_format_kwarg_in_constructor(tmp_path, setup_path):
    msg = 'format is not a defined argument for HDFStore'
    with pytest.raises(ValueError, match=msg):
        HDFStore(tmp_path / setup_path, format='table')