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
@pytest.mark.single_cpu
def test_dont_register_by_default(self):
    code = 'import matplotlib.units; import pandas as pd; units = dict(matplotlib.units.registry); assert pd.Timestamp not in units'
    call = [sys.executable, '-c', code]
    assert subprocess.check_call(call) == 0