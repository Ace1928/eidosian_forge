from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.fixture
def lines_json_df():
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    return df.to_json(lines=True, orient='records')