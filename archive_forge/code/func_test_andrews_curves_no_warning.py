import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_andrews_curves_no_warning(self, iris):
    from pandas.plotting import andrews_curves
    df = iris
    with tm.assert_produces_warning(None):
        _check_plot_works(andrews_curves, frame=df, class_column='Name')