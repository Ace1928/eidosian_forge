import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_all_keyword_arguments():
    with tm.assert_produces_warning(None):
        assert h(a=1, b=2) == 3