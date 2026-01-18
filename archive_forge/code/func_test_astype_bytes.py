import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_bytes(self):
    result = DataFrame(['foo', 'bar', 'baz']).astype(bytes)
    assert result.dtypes[0] == np.dtype('S3')