import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_four_arguments():
    with tm.assert_produces_warning(FutureWarning):
        assert f(1, 2, 3, 4) == 10