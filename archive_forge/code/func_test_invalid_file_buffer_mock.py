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
def test_invalid_file_buffer_mock(all_parsers):
    parser = all_parsers
    msg = 'Invalid file path or buffer object type'

    class Foo:
        pass
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(Foo())