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
def test_header_and_one_column(self, flavor_read_html):
    """
        Don't fail with bs4 when there is a header and only one column
        as described in issue #9178
        """
    result = flavor_read_html(StringIO('<table>\n                <thead>\n                    <tr>\n                        <th>Header</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>first</td>\n                    </tr>\n                </tbody>\n            </table>'))[0]
    expected = DataFrame(data={'Header': 'first'}, index=[0])
    tm.assert_frame_equal(result, expected)