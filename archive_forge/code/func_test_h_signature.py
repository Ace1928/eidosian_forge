import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_h_signature():
    assert str(inspect.signature(h)) == '(*, a=0, b=0, c=0, d=0)'