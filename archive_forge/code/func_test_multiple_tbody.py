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
def test_multiple_tbody(self, flavor_read_html):
    result = flavor_read_html(StringIO('<table>\n            <thead>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>1</td>\n                    <td>2</td>\n                </tr>\n            </tbody>\n            <tbody>\n                <tr>\n                    <td>3</td>\n                    <td>4</td>\n                </tr>\n            </tbody>\n        </table>'))[0]
    expected = DataFrame(data=[[1, 2], [3, 4]], columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)