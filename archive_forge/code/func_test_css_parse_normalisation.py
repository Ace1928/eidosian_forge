import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
@pytest.mark.parametrize('name,norm,abnorm', [('whitespace', 'hello: world; foo: bar', ' \t hello \t :\n  world \n  ;  \n foo: \tbar\n\n'), ('case', 'hello: world; foo: bar', 'Hello: WORLD; foO: bar'), ('empty-decl', 'hello: world; foo: bar', '; hello: world;; foo: bar;\n; ;'), ('empty-list', '', ';')])
def test_css_parse_normalisation(name, norm, abnorm):
    assert_same_resolution(norm, abnorm)