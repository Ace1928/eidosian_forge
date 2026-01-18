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
@pytest.fixture(params=[pytest.param('bs4', marks=[td.skip_if_no('bs4'), td.skip_if_no('html5lib')]), pytest.param('lxml', marks=td.skip_if_no('lxml'))])
def flavor_read_html(request):
    return partial(read_html, flavor=request.param)