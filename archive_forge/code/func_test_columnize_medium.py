import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
@pytest.mark.parametrize('row_first', [True, False])
def test_columnize_medium(row_first):
    """Test with inputs than shouldn't be wider than 80"""
    size = 40
    items = [l * size for l in 'abc']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(items, row_first=row_first, displaywidth=80)
    assert out == '\n'.join(items + ['']), 'row_first={0}'.format(row_first)