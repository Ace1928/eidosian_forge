import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_filter_args_method():
    obj = Klass()
    assert filter_args(obj.f, [], (1,)) == {'x': 1, 'self': obj}