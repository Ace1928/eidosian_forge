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
def test_bs4_version_fails(monkeypatch, datapath):
    bs4 = pytest.importorskip('bs4')
    pytest.importorskip('html5lib')
    monkeypatch.setattr(bs4, '__version__', '4.2')
    with pytest.raises(ImportError, match='Pandas requires version'):
        read_html(datapath('io', 'data', 'html', 'spam.html'), flavor='bs4')