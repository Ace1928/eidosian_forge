import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_foo_signature():
    assert str(inspect.signature(Foo.baz)) == '(self, bar=None, *, foobar=None)'