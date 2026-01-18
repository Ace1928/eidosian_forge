import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@parametrize('func,args,filtered_args', [(f, [[], (1,)], {'x': 1, 'y': 0}), (f, [['x'], (1,)], {'y': 0}), (f, [['y'], (0,)], {'x': 0}), (f, [['y'], (0,), {'y': 1}], {'x': 0}), (f, [['x', 'y'], (0,)], {}), (f, [[], (0,), {'y': 1}], {'x': 0, 'y': 1}), (f, [['y'], (), {'x': 2, 'y': 1}], {'x': 2}), (g, [[], (), {'x': 1}], {'x': 1}), (i, [[], (2,)], {'x': 2})])
def test_filter_args(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args