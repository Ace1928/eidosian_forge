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
@skip_pyarrow
def test_file_descriptor_leak(all_parsers, using_copy_on_write):
    parser = all_parsers
    with tm.ensure_clean() as path:
        with pytest.raises(EmptyDataError, match='No columns to parse from file'):
            parser.read_csv(path)