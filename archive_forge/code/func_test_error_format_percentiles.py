from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('percentiles', [[0.1, np.nan, 0.5], [-0.001, 0.1, 0.5], [2, 0.1, 0.5], [0.1, 0.5, 'a']])
def test_error_format_percentiles(self, percentiles):
    msg = 'percentiles should all be in the interval \\[0,1\\]'
    with pytest.raises(ValueError, match=msg):
        fmt.format_percentiles(percentiles)