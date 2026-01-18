import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_columnize_random():
    """Test with random input to hopefully catch edge case """
    for row_first in [True, False]:
        for nitems in [random.randint(2, 70) for i in range(2, 20)]:
            displaywidth = random.randint(20, 200)
            rand_len = [random.randint(2, displaywidth) for i in range(nitems)]
            items = ['x' * l for l in rand_len]
            with pytest.warns(PendingDeprecationWarning):
                out = text.columnize(items, row_first=row_first, displaywidth=displaywidth)
            longer_line = max([len(x) for x in out.split('\n')])
            longer_element = max(rand_len)
            assert longer_line <= displaywidth, f'Columnize displayed something lager than displaywidth : {longer_line}\nlonger element : {longer_element}\ndisplaywidth : {displaywidth}\nnumber of element : {nitems}\nsize of each element : {rand_len}\nrow_first={row_first}\n'