import shlex
import subprocess
import time
import uuid
import pytest
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv
@pytest.fixture(params=['python', pytest.param('pyarrow', marks=td.skip_if_no('pyarrow'))])
def string_storage(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param