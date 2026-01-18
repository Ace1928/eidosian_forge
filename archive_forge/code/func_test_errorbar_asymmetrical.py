from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_errorbar_asymmetrical(self):
    err = np.random.default_rng(2).random((3, 2, 5))
    df = DataFrame(np.arange(15).reshape(3, 5)).T
    ax = df.plot(yerr=err, xerr=err / 2)
    yerr_0_0 = ax.collections[1].get_paths()[0].vertices[:, 1]
    expected_0_0 = err[0, :, 0] * np.array([-1, 1])
    tm.assert_almost_equal(yerr_0_0, expected_0_0)
    msg = re.escape('Asymmetrical error bars should be provided with the shape (3, 2, 5)')
    with pytest.raises(ValueError, match=msg):
        df.plot(yerr=err.T)