import shlex
import subprocess
import time
import uuid
import pytest
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv
@pytest.fixture
def feather_file(datapath):
    return datapath('io', 'data', 'feather', 'feather-0_3_1.feather')