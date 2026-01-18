import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_func_code_consistency():
    from joblib.parallel import Parallel, delayed
    codes = Parallel(n_jobs=2)((delayed(_get_code)() for _ in range(5)))
    assert len(set(codes)) == 1