import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
def assert_same_resolution(css1, css2, inherited=None):
    resolve = CSSResolver()
    resolved1 = resolve(css1, inherited=inherited)
    resolved2 = resolve(css2, inherited=inherited)
    assert resolved1 == resolved2