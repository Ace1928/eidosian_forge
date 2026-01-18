import sys
import dill
def test_data_changed():
    FooS = dill.copy(Foo)
    try:
        f = FooS()
        f.data = 1024
        res = f.data
    except Exception:
        e = sys.exc_info()[1]
        raise AssertionError(str(e))
    else:
        assert res == 1024