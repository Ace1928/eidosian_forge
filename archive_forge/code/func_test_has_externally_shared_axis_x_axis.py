import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_has_externally_shared_axis_x_axis(self):
    func = plotting._matplotlib.tools._has_externally_shared_axis
    fig = mpl.pyplot.figure()
    plots = fig.subplots(2, 4)
    plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
    plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])
    plots[0][1].twinx()
    plots[0][2].twinx()
    assert func(plots[0][0], 'x')
    assert not func(plots[0][1], 'x')
    assert func(plots[0][2], 'x')
    assert not func(plots[0][3], 'x')