import datetime
import dateutil.tz
import dateutil.rrule
import functools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import rc_context, style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.ticker as mticker
def test_julian2num():
    mdates._reset_epoch_test_example()
    mdates.set_epoch('0000-12-31')
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mdates.julian2num(2440588.5) == 719164.0
        assert mdates.num2julian(719165.0) == 2440589.5
    mdates._reset_epoch_test_example()
    mdates.set_epoch('1970-01-01T00:00:00')
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mdates.julian2num(2440588.5) == 1.0
        assert mdates.num2julian(2.0) == 2440589.5