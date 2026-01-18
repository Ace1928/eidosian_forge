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
def test_extract_links_bad(self, spam_data):
    msg = '`extract_links` must be one of {None, "header", "footer", "body", "all"}, got "incorrect"'
    with pytest.raises(ValueError, match=msg):
        read_html(spam_data, extract_links='incorrect')