import io
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('align, exp', [('left', [no_bar(), bar_to(100, 'green')]), ('right', [bar_to(100, 'red'), no_bar()]), ('mid', [bar_to(25, 'red'), bar_from_to(25, 100, 'green')]), ('zero', [bar_from_to(33.33, 50, 'red'), bar_from_to(50, 100, 'green')])])
def test_colors_mixed(align, exp):
    data = DataFrame([[-1], [3]])
    result = data.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result == {(0, 0): exp[0], (1, 0): exp[1]}