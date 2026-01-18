import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('prop, expected', [('1pt red solid', ('red', 'solid', '1pt')), ('red 1pt solid', ('red', 'solid', '1pt')), ('red solid 1pt', ('red', 'solid', '1pt')), ('solid 1pt red', ('red', 'solid', '1pt')), ('red solid', ('red', 'solid', '1.500000pt')), ('1pt solid', ('black', 'solid', '1pt')), ('1pt red', ('red', 'none', '1pt')), ('red', ('red', 'none', '1.500000pt')), ('1pt', ('black', 'none', '1pt')), ('solid', ('black', 'solid', '1.500000pt')), ('1em', ('black', 'none', '12pt'))])
def test_css_border_shorthands(prop, expected):
    color, style, width = expected
    assert_resolves(f'border-left: {prop}', {'border-left-color': color, 'border-left-style': style, 'border-left-width': width})