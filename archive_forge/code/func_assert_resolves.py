import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
def assert_resolves(css, props, inherited=None):
    resolve = CSSResolver()
    actual = resolve(css, inherited=inherited)
    assert props == actual