from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('caption, cap_exp', [('full', ('{full}', '')), (('full', 'short'), ('{full}', '[short]'))])
@pytest.mark.parametrize('label, lab_exp', [(None, ''), ('tab:A', ' \\label{tab:A}')])
def test_longtable_caption_label(styler, caption, cap_exp, label, lab_exp):
    cap_exp1 = f'\\caption{cap_exp[1]}{cap_exp[0]}'
    cap_exp2 = f'\\caption[]{cap_exp[0]}'
    expected = dedent(f'        {cap_exp1}{lab_exp} \\\\\n         & A & B & C \\\\\n        \\endfirsthead\n        {cap_exp2} \\\\\n        ')
    assert expected in styler.to_latex(environment='longtable', caption=caption, label=label)