import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_has_externally_shared_axis_invalid_compare_axis(self):
    func = plotting._matplotlib.tools._has_externally_shared_axis
    fig = mpl.pyplot.figure()
    plots = fig.subplots(4, 2)
    plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])
    msg = "needs 'x' or 'y' as a second parameter"
    with pytest.raises(ValueError, match=msg):
        func(plots[0][0], 'z')