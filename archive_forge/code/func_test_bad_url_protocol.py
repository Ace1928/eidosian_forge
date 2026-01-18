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
@pytest.mark.network
@pytest.mark.single_cpu
def test_bad_url_protocol(self, httpserver, flavor_read_html):
    httpserver.serve_content('urlopen error unknown url type: git', code=404)
    with pytest.raises(URLError, match='urlopen error unknown url type: git'):
        flavor_read_html('git://github.com', match='.*Water.*')