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
def test_extract_links_all_no_header(self, flavor_read_html):
    data = "\n        <table>\n          <tr>\n            <td>\n              <a href='https://google.com'>Google.com</a>\n            </td>\n          </tr>\n        </table>\n        "
    result = flavor_read_html(StringIO(data), extract_links='all')[0]
    expected = DataFrame([[('Google.com', 'https://google.com')]])
    tm.assert_frame_equal(result, expected)