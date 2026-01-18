import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_ndframe_getitem_caching_issue(self, request, using_copy_on_write, warn_copy_on_write):
    if not (using_copy_on_write or warn_copy_on_write):
        request.applymarker(pytest.mark.xfail(reason='Unclear behavior.'))
    df = pd.DataFrame({'A': [0]}).set_flags(allows_duplicate_labels=False)
    assert df['A'].flags.allows_duplicate_labels is False
    df.flags.allows_duplicate_labels = True
    assert df['A'].flags.allows_duplicate_labels is True