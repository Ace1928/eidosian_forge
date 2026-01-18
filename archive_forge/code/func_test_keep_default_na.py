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
def test_keep_default_na(self, flavor_read_html):
    html_data = '<table>\n                        <thead>\n                            <tr>\n                            <th>a</th>\n                            </tr>\n                        </thead>\n                        <tbody>\n                            <tr>\n                            <td> N/A</td>\n                            </tr>\n                            <tr>\n                            <td> NA</td>\n                            </tr>\n                        </tbody>\n                    </table>'
    expected_df = DataFrame({'a': ['N/A', 'NA']})
    html_df = flavor_read_html(StringIO(html_data), keep_default_na=False)[0]
    tm.assert_frame_equal(expected_df, html_df)
    expected_df = DataFrame({'a': [np.nan, np.nan]})
    html_df = flavor_read_html(StringIO(html_data), keep_default_na=True)[0]
    tm.assert_frame_equal(expected_df, html_df)