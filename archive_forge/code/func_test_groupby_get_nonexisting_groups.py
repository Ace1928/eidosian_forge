import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_groupby_get_nonexisting_groups():
    df = pd.DataFrame(data={'A': ['a1', 'a2', None], 'B': ['b1', 'b2', 'b1'], 'val': [1, 2, 3]})
    grps = df.groupby(by=['A', 'B'])
    msg = "('a2', 'b1')"
    with pytest.raises(KeyError, match=msg):
        grps.get_group(('a2', 'b1'))