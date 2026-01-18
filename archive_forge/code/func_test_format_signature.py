import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@parametrize('func,args,kwargs,sgn_expected', [(g, [list(range(5))], {}, 'g([0, 1, 2, 3, 4])'), (k, [1, 2, (3, 4)], {'y': True}, 'k(1, 2, (3, 4), y=True)')])
def test_format_signature(func, args, kwargs, sgn_expected):
    path, sgn_result = format_signature(func, *args, **kwargs)
    assert sgn_result == sgn_expected