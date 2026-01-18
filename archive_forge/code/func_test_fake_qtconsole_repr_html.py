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
def test_fake_qtconsole_repr_html(self, float_frame):
    df = float_frame

    def get_ipython():
        return {'config': {'KernelApp': {'parent_appname': 'ipython-qtconsole'}}}
    repstr = df._repr_html_()
    assert repstr is not None
    with option_context('display.max_rows', 5, 'display.max_columns', 2):
        repstr = df._repr_html_()
    assert 'class' in repstr