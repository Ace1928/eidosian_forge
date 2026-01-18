import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_legacy_by_fig_error(self, ts):
    fig, _ = mpl.pyplot.subplots(1, 1)
    msg = "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' instance will be created"
    with pytest.raises(ValueError, match=msg):
        ts.hist(by=ts.index, figure=fig)