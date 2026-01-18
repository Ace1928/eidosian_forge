import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
@pytest.mark.parametrize('expected, width, row_first, spread', (('aaaaa  bbbbb  ccccc  ddddd\n', 80, False, False), ('aaaaa  ccccc\nbbbbb  ddddd\n', 25, False, False), ('aaaaa  ccccc\nbbbbb  ddddd\n', 12, False, False), ('aaaaa\nbbbbb\nccccc\nddddd\n', 10, False, False), ('aaaaa  bbbbb  ccccc  ddddd\n', 80, True, False), ('aaaaa  bbbbb\nccccc  ddddd\n', 25, True, False), ('aaaaa  bbbbb\nccccc  ddddd\n', 12, True, False), ('aaaaa\nbbbbb\nccccc\nddddd\n', 10, True, False), ('aaaaa      bbbbb      ccccc      ddddd\n', 40, False, True), ('aaaaa          ccccc\nbbbbb          ddddd\n', 20, False, True), ('aaaaa  ccccc\nbbbbb  ddddd\n', 12, False, True), ('aaaaa\nbbbbb\nccccc\nddddd\n', 10, False, True)))
def test_columnize(expected, width, row_first, spread):
    """Basic columnize tests."""
    size = 5
    items = [l * size for l in 'abcd']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(items, displaywidth=width, row_first=row_first, spread=spread)
        assert out == expected