import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('size,relative_to,resolved', [('1em', None, '12pt'), ('1.0em', None, '12pt'), ('1.25em', None, '15pt'), ('1em', '16pt', '16pt'), ('1.0em', '16pt', '16pt'), ('1.25em', '16pt', '20pt'), ('1rem', '16pt', '12pt'), ('1.0rem', '16pt', '12pt'), ('1.25rem', '16pt', '15pt'), ('100%', None, '12pt'), ('125%', None, '15pt'), ('100%', '16pt', '16pt'), ('125%', '16pt', '20pt'), ('2ex', None, '12pt'), ('2.0ex', None, '12pt'), ('2.50ex', None, '15pt'), ('inherit', '16pt', '16pt'), ('smaller', None, '10pt'), ('smaller', '18pt', '15pt'), ('larger', None, f'{14.4:f}pt'), ('larger', '15pt', '18pt')])
def test_css_relative_font_size(size, relative_to, resolved):
    if relative_to is None:
        inherited = None
    else:
        inherited = {'font-size': relative_to}
    assert_resolves(f'font-size: {size}', {'font-size': resolved}, inherited=inherited)