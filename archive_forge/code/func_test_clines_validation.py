from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('clines', ['bad', 'index', 'skip-last', 'all', 'data'])
def test_clines_validation(clines, styler):
    msg = f'`clines` value of {clines} is invalid.'
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(clines=clines)