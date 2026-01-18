import pytest
from pandas.util._validators import validate_args_and_kwargs
@pytest.mark.parametrize('args,kwargs', [((), {'foo': -5, 'bar': 2}), ((-5, 2), {})])
def test_missing_args_or_kwargs(args, kwargs, _fname):
    bad_arg = 'bar'
    min_fname_arg_count = 2
    compat_args = {'foo': -5, bad_arg: 1}
    msg = f"the '{bad_arg}' parameter is not supported in the pandas implementation of {_fname}\\(\\)"
    with pytest.raises(ValueError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)