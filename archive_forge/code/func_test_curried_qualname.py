from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_curried_qualname():

    def preserves_identity(obj):
        return pickle.loads(pickle.dumps(obj)) is obj
    assert preserves_identity(GlobalCurried)
    assert preserves_identity(GlobalCurried.func.f1)
    assert preserves_identity(GlobalCurried.func.NestedCurried)
    assert preserves_identity(GlobalCurried.func.NestedCurried.func.f2)
    assert preserves_identity(GlobalCurried.func.Nested.f3)
    global_curried1 = GlobalCurried(1)
    global_curried2 = pickle.loads(pickle.dumps(global_curried1))
    assert global_curried1 is not global_curried2
    assert global_curried1(2).f1(3, 4) == global_curried2(2).f1(3, 4) == 10
    global_curried3 = global_curried1(2)
    global_curried4 = pickle.loads(pickle.dumps(global_curried3))
    assert global_curried3 is not global_curried4
    assert global_curried3.f1(3, 4) == global_curried4.f1(3, 4) == 10
    func1 = global_curried1(2).f1(3)
    func2 = pickle.loads(pickle.dumps(func1))
    assert func1 is not func2
    assert func1(4) == func2(4) == 10
    nested_curried1 = GlobalCurried.func.NestedCurried(1)
    nested_curried2 = pickle.loads(pickle.dumps(nested_curried1))
    assert nested_curried1 is not nested_curried2
    assert nested_curried1(2).f2(3, 4) == nested_curried2(2).f2(3, 4) == 10