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
def salaries_table(datapath):
    """DataFrame with the salaries dataset"""
    return read_csv(datapath('io', 'parser', 'data', 'salaries.csv'), sep='\t')