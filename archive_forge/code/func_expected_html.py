from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def expected_html(datapath, name):
    """
    Read HTML file from formats data directory.

    Parameters
    ----------
    datapath : pytest fixture
        The datapath fixture injected into a test by pytest.
    name : str
        The name of the HTML file without the suffix.

    Returns
    -------
    str : contents of HTML file.
    """
    filename = '.'.join([name, 'html'])
    filepath = datapath('io', 'formats', 'data', 'html', filename)
    with open(filepath, encoding='utf-8') as f:
        html = f.read()
    return html.rstrip()