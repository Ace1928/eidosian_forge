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
@pytest.mark.parametrize('displayed_only,exp0,exp1', [(True, DataFrame(['foo']), None), (False, DataFrame(['foo  bar  baz  qux']), DataFrame(['foo']))])
def test_displayed_only(self, displayed_only, exp0, exp1, flavor_read_html):
    data = '<html>\n          <body>\n            <table>\n              <tr>\n                <td>\n                  foo\n                  <span style="display:none;text-align:center">bar</span>\n                  <span style="display:none">baz</span>\n                  <span style="display: none">qux</span>\n                </td>\n              </tr>\n            </table>\n            <table style="display: none">\n              <tr>\n                <td>foo</td>\n              </tr>\n            </table>\n          </body>\n        </html>'
    dfs = flavor_read_html(StringIO(data), displayed_only=displayed_only)
    tm.assert_frame_equal(dfs[0], exp0)
    if exp1 is not None:
        tm.assert_frame_equal(dfs[1], exp1)
    else:
        assert len(dfs) == 1