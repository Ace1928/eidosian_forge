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
def test_decimal_rows(self, flavor_read_html):
    result = flavor_read_html(StringIO('<html>\n            <body>\n             <table>\n                <thead>\n                    <tr>\n                        <th>Header</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1100#101</td>\n                    </tr>\n                </tbody>\n            </table>\n            </body>\n        </html>'), decimal='#')[0]
    expected = DataFrame(data={'Header': 1100.101}, index=[0])
    assert result['Header'].dtype == np.dtype('float64')
    tm.assert_frame_equal(result, expected)