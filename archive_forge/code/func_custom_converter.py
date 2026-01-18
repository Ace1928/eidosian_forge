import contextlib
import time
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter
def custom_converter(css):
    return {'font': {'color': {'rgb': '111222'}}}