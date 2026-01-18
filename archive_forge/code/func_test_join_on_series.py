import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_series(self, target_source):
    target, source = target_source
    result = target.join(source['MergedA'], on='C')
    expected = target.join(source[['MergedA']], on='C')
    tm.assert_frame_equal(result, expected)