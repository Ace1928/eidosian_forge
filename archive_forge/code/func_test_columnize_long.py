import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
@pytest.mark.parametrize('row_first', [True, False])
def test_columnize_long(row_first):
    """Test columnize with inputs longer than the display window"""
    size = 11
    items = [l * size for l in 'abc']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(items, row_first=row_first, displaywidth=size - 1)
    assert out == '\n'.join(items + ['']), 'row_first={0}'.format(row_first)