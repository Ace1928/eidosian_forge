import shlex
import subprocess
import time
import uuid
import pytest
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv
@pytest.fixture(params=_compression_formats_params)
def compression_ext(request):
    return request.param[0]