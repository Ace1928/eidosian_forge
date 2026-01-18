from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.usefixtures('pyarrow_xfail')
def test_delimiter_with_usecols_and_parse_dates(all_parsers):
    result = all_parsers.read_csv(StringIO('"dump","-9,1","-9,1",20101010'), engine='python', names=['col', 'col1', 'col2', 'col3'], usecols=['col1', 'col2', 'col3'], parse_dates=['col3'], decimal=',')
    expected = DataFrame({'col1': [-9.1], 'col2': [-9.1], 'col3': [Timestamp('2010-10-10')]})
    tm.assert_frame_equal(result, expected)