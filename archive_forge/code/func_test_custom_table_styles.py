from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_custom_table_styles(styler):
    styler.set_table_styles([{'selector': 'mycommand', 'props': ':{myoptions}'}, {'selector': 'mycommand2', 'props': ':{myoptions2}'}])
    expected = dedent('        \\begin{table}\n        \\mycommand{myoptions}\n        \\mycommand2{myoptions2}\n        ')
    assert expected in styler.to_latex()