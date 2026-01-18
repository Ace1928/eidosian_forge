import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('objs, kwargs', [([pd.Series(1, index=[0, 1], name='a'), pd.Series(2, index=[0, 1], name='a')], {'axis': 1})])
def test_concat_raises(self, objs, kwargs):
    objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.concat(objs, **kwargs)