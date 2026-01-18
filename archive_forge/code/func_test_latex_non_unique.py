from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_latex_non_unique(styler):
    result = styler.to_latex()
    assert result == dedent('        \\begin{tabular}{lrrr}\n         & c & d & d \\\\\n        i & 1.000000 & 2.000000 & 3.000000 \\\\\n        j & 4.000000 & 5.000000 & 6.000000 \\\\\n        j & 7.000000 & 8.000000 & 9.000000 \\\\\n        \\end{tabular}\n    ')