import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@parametrize('func,args,filtered_args', [(k, [[], (1, 2), {'ee': 2}], {'*': [1, 2], '**': {'ee': 2}}), (k, [[], (3, 4)], {'*': [3, 4], '**': {}})] + test_filter_kwargs_extra_params)
def test_filter_kwargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args