import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.single_cpu
def test_missing_required_dependency():
    pyexe = sys.executable.replace('\\', '/')
    call = [pyexe, '-c', 'import pandas;print(pandas.__file__)']
    output = subprocess.check_output(call).decode()
    if 'site-packages' in output:
        pytest.skip('pandas installed as site package')
    call = [pyexe, '-sSE', '-c', 'import pandas']
    msg = f"Command '\\['{pyexe}', '-sSE', '-c', 'import pandas'\\]' returned non-zero exit status 1."
    with pytest.raises(subprocess.CalledProcessError, match=msg) as exc:
        subprocess.check_output(call, stderr=subprocess.STDOUT)
    output = exc.value.stdout.decode()
    for name in ['numpy', 'pytz', 'dateutil']:
        assert name in output