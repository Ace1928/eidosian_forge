import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_filter_args_edge_cases():
    assert filter_args(func_with_kwonly_args, [], (1, 2), {'kw1': 3, 'kw2': 4}) == {'a': 1, 'b': 2, 'kw1': 3, 'kw2': 4}
    with raises(ValueError) as excinfo:
        filter_args(func_with_kwonly_args, [], (1, 2, 3), {'kw2': 2})
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional parameter")
    assert filter_args(func_with_kwonly_args, ['b', 'kw2'], (1, 2), {'kw1': 3, 'kw2': 4}) == {'a': 1, 'kw1': 3}
    assert filter_args(func_with_signature, ['b'], (1, 2)) == {'a': 1}