from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_longtable_comprehensive(styler):
    result = styler.to_latex(environment='longtable', hrules=True, label='fig:A', caption=('full', 'short'))
    expected = dedent('        \\begin{longtable}{lrrl}\n        \\caption[short]{full} \\label{fig:A} \\\\\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        \\endfirsthead\n        \\caption[]{full} \\\\\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        \\endhead\n        \\midrule\n        \\multicolumn{4}{r}{Continued on next page} \\\\\n        \\midrule\n        \\endfoot\n        \\bottomrule\n        \\endlastfoot\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{longtable}\n    ')
    assert result == expected