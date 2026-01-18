from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_cell_styling(styler):
    styler.highlight_max(props='itshape:;Huge:--wrap;')
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\\n        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\\n        \\end{tabular}\n        ')
    assert expected == styler.to_latex()