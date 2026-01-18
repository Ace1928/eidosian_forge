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
def assert_equal_cell_styles(cell1, cell2):
    assert cell1.alignment.__dict__ == cell2.alignment.__dict__
    assert cell1.border.__dict__ == cell2.border.__dict__
    assert cell1.fill.__dict__ == cell2.fill.__dict__
    assert cell1.font.__dict__ == cell2.font.__dict__
    assert cell1.number_format == cell2.number_format
    assert cell1.protection.__dict__ == cell2.protection.__dict__