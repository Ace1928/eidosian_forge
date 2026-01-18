import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_func_name_on_inner_func(cached_func):
    assert get_func_name(cached_func)[1] == 'cached_func_inner'