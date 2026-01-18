import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('cls, data', [(pd.Series, np.array([])), (pd.Series, [1, 2]), (pd.DataFrame, {}), (pd.DataFrame, {'A': [1, 2]})])
def test_construction_ok(self, cls, data):
    result = cls(data)
    assert result.flags.allows_duplicate_labels is True
    result = cls(data).set_flags(allows_duplicate_labels=False)
    assert result.flags.allows_duplicate_labels is False