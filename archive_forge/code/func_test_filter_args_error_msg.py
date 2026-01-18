import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
@parametrize('exception,regex,func,args', [(ValueError, 'ignore_lst must be a list of parameters to ignore', f, ['bar', (None,)]), (ValueError, "Ignore list: argument \\'(.*)\\' is not defined", g, [['bar'], (None,)]), (ValueError, 'Wrong number of arguments', h, [[]])])
def test_filter_args_error_msg(exception, regex, func, args):
    """ Make sure that filter_args returns decent error messages, for the
        sake of the user.
    """
    with raises(exception) as excinfo:
        filter_args(func, *args)
    excinfo.match(regex)