from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input_array, expected', [({'A': ['a']}, 'A\na'), ({'A': ['a', 'b'], 'B': ['c', 'dd']}, 'A  B\na  c\nb dd'), ({'A': ['a', 1], 'B': ['aa', 1]}, 'A  B\na aa\n1  1')])
def test_format_remove_leading_space_dataframe(self, input_array, expected):
    df = DataFrame(input_array).to_string(index=False)
    assert df == expected