import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
@pytest.mark.parametrize('message, match', [('', None), ('', ''), ('Warning message', '.*'), ('Warning message', 'War'), ('Warning message', '[Ww]arning'), ('Warning message', 'age'), ('Warning message', 'age$'), ('Message 12-234 with numbers', '\\d{2}-\\d{3}'), ('Message 12-234 with numbers', '^Mes.*\\d{2}-\\d{3}'), ('Message 12-234 with numbers', '\\d{2}-\\d{3}\\s\\S+'), ('Message, which we do not match', None)])
def test_catch_warning_category_and_match(category, message, match):
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)