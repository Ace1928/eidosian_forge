import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('formatter, exp', [(lambda x: f'{x:.3f}', '9.000'), ('{:.2f}', '9.00'), ({0: '{:.1f}'}, '9.0'), (None, '9')])
def test_formatter_options_validator(formatter, exp):
    df = DataFrame([[9]])
    with option_context('styler.format.formatter', formatter):
        assert f' {exp} ' in df.style.to_latex()