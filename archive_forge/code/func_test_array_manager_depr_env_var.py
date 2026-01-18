import os
import subprocess
import sys
import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
import pandas._testing as tm
from pandas.core.internals import (
@pytest.mark.single_cpu
@pytest.mark.parametrize('manager', ['block', 'array'])
def test_array_manager_depr_env_var(manager):
    test_env = os.environ.copy()
    test_env['PANDAS_DATA_MANAGER'] = manager
    response = subprocess.run([sys.executable, '-c', 'import pandas'], capture_output=True, env=test_env, check=True)
    msg = 'FutureWarning: The env variable PANDAS_DATA_MANAGER is set'
    stderr_msg = response.stderr.decode('utf-8')
    assert msg in stderr_msg, stderr_msg