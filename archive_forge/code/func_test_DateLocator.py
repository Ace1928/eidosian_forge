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
def test_DateLocator():
    locator = mdates.DateLocator()
    assert locator.nonsingular(0, np.inf) == (0, 1)
    assert locator.nonsingular(0, 1) == (0, 1)
    assert locator.nonsingular(1, 0) == (0, 1)
    assert locator.nonsingular(0, 0) == (-2, 2)
    locator.create_dummy_axis()
    assert locator.datalim_to_dt() == (datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc), datetime.datetime(1970, 1, 2, 0, 0, tzinfo=datetime.timezone.utc))
    assert locator.tz == mdates.UTC
    tz_str = 'Iceland'
    iceland_tz = dateutil.tz.gettz(tz_str)
    assert locator.tz != iceland_tz
    locator.set_tzinfo('Iceland')
    assert locator.tz == iceland_tz
    locator.create_dummy_axis()
    locator.axis.set_data_interval(*mdates.date2num(['2022-01-10', '2022-01-08']))
    assert locator.datalim_to_dt() == (datetime.datetime(2022, 1, 8, 0, 0, tzinfo=iceland_tz), datetime.datetime(2022, 1, 10, 0, 0, tzinfo=iceland_tz))
    plt.rcParams['timezone'] = tz_str
    locator = mdates.DateLocator()
    assert locator.tz == iceland_tz
    with pytest.raises(ValueError, match='Aiceland is not a valid timezone'):
        mdates.DateLocator(tz='Aiceland')
    with pytest.raises(TypeError, match='tz must be string or tzinfo subclass.'):
        mdates.DateLocator(tz=1)