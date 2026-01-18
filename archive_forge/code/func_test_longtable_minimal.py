from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_longtable_minimal(styler):
    result = styler.to_latex(environment='longtable')
    expected = dedent('        \\begin{longtable}{lrrl}\n         & A & B & C \\\\\n        \\endfirsthead\n         & A & B & C \\\\\n        \\endhead\n        \\multicolumn{4}{r}{Continued on next page} \\\\\n        \\endfoot\n        \\endlastfoot\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{longtable}\n    ')
    assert result == expected