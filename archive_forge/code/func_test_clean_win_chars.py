import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_clean_win_chars():
    string = 'C:\\foo\\bar\\main.py'
    mangled_string = _clean_win_chars(string)
    for char in ('\\', ':', '<', '>', '!'):
        assert char not in mangled_string