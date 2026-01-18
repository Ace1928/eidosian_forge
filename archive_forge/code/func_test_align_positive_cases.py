import io
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('align, exp', [('left', [no_bar(), bar_to(50), bar_to(100)]), ('right', [bar_to(100), bar_from_to(50, 100), no_bar()]), ('mid', [bar_to(33.33), bar_to(66.66), bar_to(100)]), ('zero', [bar_from_to(50, 66.7), bar_from_to(50, 83.3), bar_from_to(50, 100)]), ('mean', [bar_to(50), no_bar(), bar_from_to(50, 100)]), (2.0, [bar_to(50), no_bar(), bar_from_to(50, 100)]), (np.median, [bar_to(50), no_bar(), bar_from_to(50, 100)])])
def test_align_positive_cases(df_pos, align, exp):
    result = df_pos.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected