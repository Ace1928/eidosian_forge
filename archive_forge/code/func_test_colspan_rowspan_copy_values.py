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
def test_colspan_rowspan_copy_values(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td colspan="2">X</td>\n                    <td>Y</td>\n                    <td rowspan="2">Z</td>\n                    <td>W</td>\n                </tr>\n                <tr>\n                    <td>A</td>\n                    <td colspan="2">B</td>\n                    <td>C</td>\n                </tr>\n            </table>\n        '), header=0)[0]
    expected = DataFrame(data=[['A', 'B', 'B', 'Z', 'C']], columns=['X', 'X.1', 'Y', 'Z', 'W'])
    tm.assert_frame_equal(result, expected)