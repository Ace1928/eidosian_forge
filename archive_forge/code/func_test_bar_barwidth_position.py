import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('kwargs', [{'kind': 'bar', 'stacked': False}, {'kind': 'bar', 'stacked': True}, {'kind': 'barh', 'stacked': False}, {'kind': 'barh', 'stacked': True}, {'kind': 'bar', 'subplots': True}, {'kind': 'barh', 'subplots': True}])
def test_bar_barwidth_position(self, kwargs):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)