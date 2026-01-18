import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('histtype, expected', [('bar', True), ('barstacked', True), ('step', False), ('stepfilled', True)])
def test_histtype_argument(self, histtype, expected):
    df = DataFrame(np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=['a', 'b'])
    ax = df.hist(by='a', histtype=histtype)
    _check_patches_all_filled(ax, filled=expected)