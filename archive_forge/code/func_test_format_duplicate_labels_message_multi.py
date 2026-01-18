import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_format_duplicate_labels_message_multi():
    idx = pd.MultiIndex.from_product([['A'], ['a', 'b', 'a', 'b', 'c']])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame({'positions': [[0, 2], [1, 3]]}, index=pd.MultiIndex.from_product([['A'], ['a', 'b']]))
    tm.assert_frame_equal(result, expected)