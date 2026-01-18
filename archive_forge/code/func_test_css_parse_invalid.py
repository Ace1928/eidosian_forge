import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('invalid_css,remainder', [('hello-world', ''), ('border-style: solid; hello-world', 'border-style: solid'), ('border-style: solid; hello-world; font-weight: bold', 'border-style: solid; font-weight: bold'), ('font-size: blah', 'font-size: 1em'), ('font-size: 1a2b', 'font-size: 1em'), ('font-size: 1e5pt', 'font-size: 1em'), ('font-size: 1+6pt', 'font-size: 1em'), ('font-size: 1unknownunit', 'font-size: 1em'), ('font-size: 10', 'font-size: 1em'), ('font-size: 10 pt', 'font-size: 1em'), ('border-top: 1pt solid red green', 'border-top: 1pt solid green')])
def test_css_parse_invalid(invalid_css, remainder):
    with tm.assert_produces_warning(CSSWarning):
        assert_same_resolution(invalid_css, remainder)