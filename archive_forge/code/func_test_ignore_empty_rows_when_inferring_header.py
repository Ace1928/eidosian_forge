from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_ignore_empty_rows_when_inferring_header(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <thead>\n                    <tr><th></th><th></tr>\n                    <tr><th>A</th><th>B</th></tr>\n                    <tr><th>a</th><th>b</th></tr>\n                </thead>\n                <tbody>\n                    <tr><td>1</td><td>2</td></tr>\n                </tbody>\n            </table>\n        '))[0]
    columns = MultiIndex(levels=[['A', 'B'], ['a', 'b']], codes=[[0, 1], [0, 1]])
    expected = DataFrame(data=[[1, 2]], columns=columns)
    tm.assert_frame_equal(result, expected)