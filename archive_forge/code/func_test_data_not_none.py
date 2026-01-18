import sys
import dill
def test_data_not_none():
    FooS = dill.copy(Foo)
    assert FooS.data.fget is not None
    assert FooS.data.fset is not None
    assert FooS.data.fdel is None