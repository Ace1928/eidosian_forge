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
@pytest.mark.slow
def test_gold_canyon(self, banklist_data, flavor_read_html):
    gc = 'Gold Canyon'
    with open(banklist_data, encoding='utf-8') as f:
        raw_text = f.read()
    assert gc in raw_text
    df = flavor_read_html(banklist_data, match='Gold Canyon', attrs={'id': 'table'})[0]
    assert gc in df.to_string()