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
def test_empty_tables(self, flavor_read_html):
    """
        Make sure that read_html ignores empty tables.
        """
    html = '\n            <table>\n                <thead>\n                    <tr>\n                        <th>A</th>\n                        <th>B</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1</td>\n                        <td>2</td>\n                    </tr>\n                </tbody>\n            </table>\n            <table>\n                <tbody>\n                </tbody>\n            </table>\n        '
    result = flavor_read_html(StringIO(html))
    assert len(result) == 1