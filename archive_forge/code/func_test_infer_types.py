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
def test_infer_types(self, spam_data, flavor_read_html):
    df1 = flavor_read_html(spam_data, match='.*Water.*', index_col=0)
    df2 = flavor_read_html(spam_data, match='Unit', index_col=0)
    assert_framelist_equal(df1, df2)