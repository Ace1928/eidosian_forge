import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_get_datetime64_unit(install_temp):
    import checks
    dt64 = np.datetime64('2016-01-01', 'ns')
    result = checks.get_dt64_unit(dt64)
    expected = 10
    assert result == expected
    td64 = np.timedelta64(12345, 'h')
    result = checks.get_dt64_unit(td64)
    expected = 5
    assert result == expected