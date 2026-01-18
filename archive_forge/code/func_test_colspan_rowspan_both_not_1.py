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
def test_colspan_rowspan_both_not_1(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td rowspan="2">A</td>\n                    <td rowspan="2" colspan="3">B</td>\n                    <td>C</td>\n                </tr>\n                <tr>\n                    <td>D</td>\n                </tr>\n            </table>\n        '), header=0)[0]
    expected = DataFrame(data=[['A', 'B', 'B', 'B', 'D']], columns=['A', 'B', 'B.1', 'B.2', 'C'])
    tm.assert_frame_equal(result, expected)