from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_file_handles_with_open(all_parsers, csv1):
    parser = all_parsers
    for mode in ['r', 'rb']:
        with open(csv1, mode, encoding='utf-8' if mode == 'r' else None) as f:
            parser.read_csv(f)
            assert not f.closed