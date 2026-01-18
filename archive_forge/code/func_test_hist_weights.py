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
@pytest.mark.parametrize('weights', [0.1 * np.ones(shape=(100,)), 0.1 * np.ones(shape=(100, 2))])
def test_hist_weights(self, weights):
    df = DataFrame(dict(zip(['A', 'B'], np.random.default_rng(2).standard_normal((2, 100)))))
    ax1 = _check_plot_works(df.plot, kind='hist', weights=weights)
    ax2 = _check_plot_works(df.plot, kind='hist')
    patch_height_with_weights = [patch.get_height() for patch in ax1.patches]
    expected_patch_height = [0.1 * patch.get_height() for patch in ax2.patches]
    tm.assert_almost_equal(patch_height_with_weights, expected_patch_height)