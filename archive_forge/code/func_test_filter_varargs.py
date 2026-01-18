import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@parametrize('func,args,filtered_args', [(h, [[], (1,)], {'x': 1, 'y': 0, '*': [], '**': {}}), (h, [[], (1, 2, 3, 4)], {'x': 1, 'y': 2, '*': [3, 4], '**': {}}), (h, [[], (1, 25), {'ee': 2}], {'x': 1, 'y': 25, '*': [], '**': {'ee': 2}}), (h, [['*'], (1, 2, 25), {'ee': 2}], {'x': 1, 'y': 2, '**': {'ee': 2}})])
def test_filter_varargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args