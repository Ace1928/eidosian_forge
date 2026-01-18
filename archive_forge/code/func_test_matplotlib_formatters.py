from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
def test_matplotlib_formatters(self):
    units = pytest.importorskip('matplotlib.units')
    with cf.option_context('plotting.matplotlib.register_converters', True):
        with cf.option_context('plotting.matplotlib.register_converters', False):
            assert Timestamp not in units.registry
        assert Timestamp in units.registry