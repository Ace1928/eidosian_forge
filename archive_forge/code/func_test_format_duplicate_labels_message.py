import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_format_duplicate_labels_message():
    idx = pd.Index(['a', 'b', 'a', 'b', 'c'])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame({'positions': [[0, 2], [1, 3]]}, index=pd.Index(['a', 'b'], name='label'))
    tm.assert_frame_equal(result, expected)