from io import (
import pytest
import pandas as pd
import pandas._testing as tm
def test_no_buf():
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown()
    assert result == '|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'