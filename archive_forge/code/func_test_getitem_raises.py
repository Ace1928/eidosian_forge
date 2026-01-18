import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('getter, target', [(operator.itemgetter(['A', 'A']), None), (operator.itemgetter(['a', 'a']), 'loc'), pytest.param(operator.itemgetter(('a', ['A', 'A'])), 'loc'), (operator.itemgetter((['a', 'a'], 'A')), 'loc'), (operator.itemgetter([0, 0]), 'iloc'), pytest.param(operator.itemgetter((0, [0, 0])), 'iloc'), pytest.param(operator.itemgetter(([0, 0], 0)), 'iloc')])
def test_getitem_raises(self, getter, target):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False)
    if target:
        target = getattr(df, target)
    else:
        target = df
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        getter(target)